"""
Assistant tools for the dental AI agent.
"""

from __future__ import annotations
import re
import time
import asyncio
from typing import Optional, Dict, Any, Callable, Sequence, cast
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import (
    DEFAULT_TZ,
    BOOKING_TZ,
    BOOKED_STATUSES,
    DEFAULT_PHONE_REGION,
    APPOINTMENT_BUFFER_MINUTES,
    supabase,
    logger,
)

from models.state import PatientState
from livekit.agents import llm

from utils.phone_utils import (
    _normalize_phone_preserve_plus,
)
from utils.agent_flow import (
    build_time_parse_candidates,
    ensure_caller_phone_pending,
    has_date_reference,
    looks_like_phone_input,
    normalize_patient_name,
)
from services.database_service import is_slot_free_supabase, book_to_supabase
from services.scheduling_service import (
    get_duration_for_service,
    is_within_working_hours,
    get_next_available_slots,
    suggest_slots_around,
    WEEK_KEYS,
)
from services.appointment_management_service import (
    find_appointment_by_phone,
    cancel_appointment,
    reschedule_appointment,
)
from services.extraction_service import _iso, extract_name_quick, extract_reason_quick
from utils.contact_utils import parse_datetime_natural


# ============================================================================
# Module-level globals (injected by agent.py)
# ============================================================================

_GLOBAL_CLINIC_TZ: str = DEFAULT_TZ
_GLOBAL_CLINIC_INFO: Optional[Dict[str, Any]] = None
_GLOBAL_AGENT_SETTINGS: Optional[Dict[str, Any]] = None
_REFRESH_AGENT_MEMORY: Optional[Callable[[], None]] = None
_GLOBAL_SCHEDULE: Optional[Dict[str, Any]] = None


def update_global_clinic_info(
    info: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
) -> None:
    """Called by agent.py to inject the database context including timezone."""
    global _GLOBAL_CLINIC_INFO, _GLOBAL_AGENT_SETTINGS, _GLOBAL_CLINIC_TZ
    _GLOBAL_CLINIC_INFO = info or {}
    if settings:
        _GLOBAL_AGENT_SETTINGS = settings
    if info and info.get("timezone"):
        _GLOBAL_CLINIC_TZ = info["timezone"]
        logger.info(f"[TOOLS] Timezone updated to: {_GLOBAL_CLINIC_TZ}")


# ============================================================================
# Helpers
# ============================================================================

def _sanitize_tool_arg(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip()
    return None if s.lower() in ("null", "none", "") else s


def email_for_speech(email: str) -> str:
    if not email:
        return "unknown"
    return email.replace("@", " at ").replace(".", " dot ")


def contact_phase_allowed(state: PatientState) -> bool:
    safety_check = (
        state.full_name is not None
        and state.time_status == "valid"
        and state.dt_local is not None
    )
    fallback_check = getattr(state, "contact_phase_started", False) is True
    caller_id_accepted = getattr(state, "caller_id_accepted", False) is True
    return safety_check or fallback_check or caller_id_accepted


def _refresh_memory():
    if _REFRESH_AGENT_MEMORY:
        try:
            _REFRESH_AGENT_MEMORY()
        except Exception:
            pass


def _contact_reference(state: PatientState) -> str:
    if state.using_caller_number or state.confirmed_contact_number_source == "caller_id":
        return "this number"
    if state.phone_last4:
        return f"your number ending in {state.phone_last4}"
    return "your number"


def _phone_confirmation_question(state: PatientState, phone_candidate: str) -> str:
    if (state.phone_source or "").lower() == "user_spoken":
        if state.phone_last4:
            return "Is this the right number to send your confirmation to?"
        return "Would you like me to use this number for appointment-related updates?"
    return "Can I use the number you're calling from for your appointment confirmation and reminders?"


def _booking_sentence(state: PatientState) -> str:
    dt = state.dt_local
    if not dt:
        return "You're all set for your appointment."
    day = dt.strftime("%A, %B %d")
    time_str = dt.strftime("%I:%M %p").lstrip("0")
    reason = state.reason or "your appointment"
    if state.full_name:
        return f"{state.full_name}, you're all set for your {reason} on {day} at {time_str}."
    return f"You're all set for your {reason} on {day} at {time_str}."


def _delivery_question_text(state: PatientState) -> str:
    target = _contact_reference(state)
    if target == "this number":
        return "I'll send your confirmation to this number. Would you like that on WhatsApp, or by SMS on this number?"
    return f"I'll send your confirmation to {target}. Would you like that on WhatsApp, or by SMS instead?"


def _apply_delivery_preference(state: PatientState, channel: str) -> str:
    normalized = str(channel).strip().lower()
    if normalized not in {"whatsapp", "sms"}:
        raise ValueError(f"Unsupported delivery channel: {channel}")

    state.delivery_channel = normalized
    state.prefers_sms = normalized == "sms"
    state.delivery_preference_pending = False
    state.delivery_preference_asked = True
    state.delivery_ask_count = 0
    logger.info(f"[TOOL] Delivery preference: {normalized}")

    if normalized == "sms":
        return "No problem, I'll send it by SMS."
    return "Perfect, I'll send it on WhatsApp."


def _date_hint_for_prompt(state: PatientState, *, fallback_text: Optional[str] = None) -> Optional[str]:
    candidate = " ".join((fallback_text or "").split()).strip()
    if candidate and has_date_reference(candidate):
        return candidate
    if state.dt_local:
        return state.dt_local.strftime("%A, %B %d")
    state_dt_text = " ".join((state.dt_text or "").split()).strip()
    if state_dt_text and has_date_reference(state_dt_text):
        return state_dt_text
    return None


def _time_reask_text(state: PatientState, *, date_hint: Optional[str] = None) -> str:
    spoken_date = _date_hint_for_prompt(state, fallback_text=date_hint)
    if spoken_date:
        lowered = spoken_date.lower()
        if lowered in {"today", "tomorrow", "day after tomorrow"}:
            return f"I didn't catch the time. What time works best {lowered}?"
        return f"I didn't catch the time. What time works best on {spoken_date}?"
    return "I didn't catch that time. What time works best for you?"


def _is_useful_time_parse_result(result: Dict[str, Any]) -> bool:
    return bool(
        result.get("datetime") is not None
        or result.get("date_only")
        or result.get("needs_clarification")
        or result.get("clarification_type")
    )


KNOWLEDGE_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "can",
    "could",
    "do",
    "for",
    "get",
    "i",
    "if",
    "in",
    "is",
    "it",
    "know",
    "like",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "tell",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "would",
    "you",
    "your",
}
PRICE_VALUE_RE = re.compile(r"\$\s?\d", re.IGNORECASE)
QUESTION_LEAD_IN_RE = re.compile(
    r"\b("
    r"what|when|where|how(?: much| long)?|"
    r"do you|can you|could you|would you|"
    r"tell me|i (?:want|would like) to know|"
    r"get to know|let me know"
    r")\b",
    re.IGNORECASE,
)
APPOINTMENT_FLOW_HINT_RE = re.compile(
    r"\b("
    r"book|booking|schedule|scheduled|appointment|reschedule|cancel|"
    r"slot|slots|available|availability|opening|openings|"
    r"day|time|tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)"
    r")\b",
    re.IGNORECASE,
)
PRICING_HINT_RE = re.compile(r"\b(price|prices|pricing|cost|costs|fee|fees|rate|rates)\b", re.IGNORECASE)
INSURANCE_HINT_RE = re.compile(r"\b(insurance|insured|coverage|covered|accept|take)\b", re.IGNORECASE)
HOURS_HINT_RE = re.compile(r"\b(hours|open|close|closing)\b", re.IGNORECASE)
LOCATION_HINT_RE = re.compile(r"\b(location|located|address)\b", re.IGNORECASE)
PARKING_HINT_RE = re.compile(r"\b(parking|park)\b", re.IGNORECASE)
PAYMENT_HINT_RE = re.compile(
    r"\b(payment|payments|pay|paid|cash|visa|mastercard|mc|amex|card|cards|financing|carecredit)\b",
    re.IGNORECASE,
)
STAFF_HINT_RE = re.compile(
    r"\b(doctor|dr\.?|dentist|provider|providers|staff|team|experience|graduated)\b",
    re.IGNORECASE,
)
POLICY_HINT_RE = re.compile(
    r"\b(policy|policies|privacy|hipaa|notice|late cancel|late cancellation|cancel fee|cancellation fee)\b",
    re.IGNORECASE,
)
EMERGENCY_HINT_RE = re.compile(
    r"\b(emergency|emergencies|urgent|same[- ]day|pain|toothache)\b",
    re.IGNORECASE,
)
LOGISTICS_HINT_RE = re.compile(r"\b(transit|metro|station|bus|subway|train|blocks?)\b", re.IGNORECASE)
SERVICE_INFO_HINT_RE = re.compile(
    r"\b(service|services|procedure|procedures|whitening|cleaning|checkup|consultation|extraction|filling|crown|root canal)\b",
    re.IGNORECASE,
)
DETAIL_REQUEST_RE = re.compile(
    r"\b(all the details|details|everything|all about|tell me about|information|info)\b",
    re.IGNORECASE,
)
SERVICE_FOLLOW_UP_HINT_RE = re.compile(
    r"\b(type|kind|option|options|better|best|difference|compare|versus|vs)\b",
    re.IGNORECASE,
)
KNOWLEDGE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|;\s+")
ANYTHING_ELSE_FOLLOW_UP_TEXT = "Is there anything else I can help you with today?"
GENERIC_SERVICE_TERMS = {
    "appointment",
    "care",
    "dental",
    "service",
    "teeth",
    "tooth",
    "treatment",
}
GENERIC_KNOWLEDGE_TERMS = {
    "accept",
    "address",
    "appointments",
    "book",
    "booking",
    "cancel",
    "cancellation",
    "card",
    "cards",
    "carecredit",
    "cash",
    "close",
    "closing",
    "cost",
    "costs",
    "coverage",
    "covered",
    "dentist",
    "doctor",
    "doctors",
    "email",
    "emergency",
    "experience",
    "fee",
    "fees",
    "financing",
    "graduated",
    "hipaa",
    "hour",
    "hours",
    "insurance",
    "located",
    "location",
    "methods",
    "metro",
    "name",
    "notice",
    "open",
    "parking",
    "park",
    "payment",
    "payments",
    "policies",
    "policy",
    "price",
    "prices",
    "pricing",
    "provider",
    "providers",
    "rate",
    "rates",
    "service",
    "services",
    "staff",
    "station",
    "team",
    "transit",
    "urgent",
    "visa",
}
SERVICE_BOUNDARY_TERMS = (
    "teeth whitening",
    "whitening",
    "root canal",
    "night guards",
    "night guard",
    "cleaning",
    "check-up",
    "checkup",
    "exam",
    "consultation",
    "consult",
    "extraction",
    "extract",
    "filling",
    "crown",
    "tooth pain",
    "toothache",
)


def _normalize_knowledge_articles(articles: Optional[Sequence[Dict[str, Any]]]) -> list[Dict[str, str]]:
    normalized: list[Dict[str, str]] = []
    for article in articles or []:
        if not isinstance(article, dict):
            continue
        title = " ".join(str(article.get("title") or "").split()).strip()
        body = " ".join(str(article.get("body") or "").split()).strip()
        category = " ".join(str(article.get("category") or "").split()).strip()
        if title or body or category:
            normalized.append({"title": title, "body": body, "category": category})
    return normalized


def _knowledge_terms(text: Optional[str]) -> set[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in KNOWLEDGE_STOPWORDS
    }


def _question_topic_terms(question: str) -> set[str]:
    terms = _knowledge_terms(question)
    service = extract_reason_quick(question)
    if service:
        terms.update(_knowledge_terms(service))
    lower = question.lower()
    if PRICING_HINT_RE.search(lower):
        terms.update({"price", "pricing", "cost", "costs", "fee", "fees", "rate", "rates"})
    if INSURANCE_HINT_RE.search(lower):
        terms.update({"insurance", "coverage", "covered", "accept", "take"})
    if HOURS_HINT_RE.search(lower):
        terms.update({"hours", "open", "close", "closing"})
    if LOCATION_HINT_RE.search(lower):
        terms.update({"location", "located", "address"})
    if PARKING_HINT_RE.search(lower):
        terms.update({"parking", "park"})
    if PAYMENT_HINT_RE.search(lower):
        terms.update({"payment", "payments", "pay", "cash", "visa", "mastercard", "amex", "card", "cards", "financing", "carecredit"})
    if STAFF_HINT_RE.search(lower):
        terms.update({"doctor", "dentist", "provider", "providers", "staff", "team"})
    if POLICY_HINT_RE.search(lower):
        terms.update({"policy", "policies", "privacy", "hipaa", "notice", "cancellation", "cancel", "fee"})
    if EMERGENCY_HINT_RE.search(lower):
        terms.update({"emergency", "urgent", "same", "day", "pain", "toothache"})
    if LOGISTICS_HINT_RE.search(lower):
        terms.update({"transit", "metro", "station", "bus", "subway", "train", "blocks"})
    if SERVICE_INFO_HINT_RE.search(lower):
        terms.update({"service", "services", "procedure", "procedures"})
    return terms


def _question_specific_terms(question: str) -> set[str]:
    return {
        term
        for term in _question_topic_terms(question)
        if term not in GENERIC_KNOWLEDGE_TERMS
    }


def _service_specific_terms(service: Optional[str]) -> set[str]:
    if not service:
        return set()
    specific_terms = {
        term
        for term in _knowledge_terms(service)
        if term not in GENERIC_SERVICE_TERMS
    }
    if specific_terms:
        return specific_terms
    fallback = " ".join(str(service or "").lower().split()).strip()
    return {fallback} if fallback else set()


def _question_with_service_context(question: str, fallback_service: Optional[str]) -> str:
    normalized = " ".join((question or "").split()).strip()
    service = " ".join(str(fallback_service or "").split()).strip()
    if not normalized or not service:
        return normalized
    if extract_reason_quick(normalized):
        return normalized

    lower_question = normalized.lower()
    lower_service = service.lower()
    if lower_service in lower_question:
        return normalized

    routed_categories = _question_knowledge_categories(normalized)
    if routed_categories.intersection({"pricing", "services"}) or SERVICE_FOLLOW_UP_HINT_RE.search(normalized):
        return f"{normalized} about {service}"
    return normalized


def _specific_knowledge_match_count(question: str, *, title: str, body: str) -> int:
    specific_terms = _question_specific_terms(question)
    if not specific_terms:
        return 0
    title_lower = title.lower()
    body_lower = body.lower()
    return sum(1 for term in specific_terms if term in title_lower or term in body_lower)


def _normalize_knowledge_category(value: Optional[str]) -> str:
    normalized = " ".join(str(value or "").strip().lower().split())
    return normalized


def _question_knowledge_categories(question: str) -> set[str]:
    lower = str(question or "").lower()
    categories: set[str] = set()
    if PRICING_HINT_RE.search(lower):
        categories.update({"pricing", "services"})
    if INSURANCE_HINT_RE.search(lower):
        categories.add("insurance")
    if HOURS_HINT_RE.search(lower):
        categories.add("hours")
    if LOCATION_HINT_RE.search(lower):
        categories.update({"location", "logistics"})
    if PARKING_HINT_RE.search(lower):
        categories.update({"parking", "location", "logistics"})
    if PAYMENT_HINT_RE.search(lower):
        categories.add("payment")
    if STAFF_HINT_RE.search(lower):
        categories.add("staff")
    if POLICY_HINT_RE.search(lower):
        categories.add("policy")
    if EMERGENCY_HINT_RE.search(lower):
        categories.add("emergency")
    if LOGISTICS_HINT_RE.search(lower):
        categories.update({"logistics", "location"})
    if SERVICE_INFO_HINT_RE.search(lower):
        categories.update({"services", "pricing"})
    return categories


def _knowledge_match_score(question: str, *, title: str, body: str, category: str = "") -> int:
    lower_question = question.lower()
    category_key = _normalize_knowledge_category(category)
    routed_categories = _question_knowledge_categories(question)
    if routed_categories and category_key and category_key not in routed_categories:
        return 0

    haystacks = f"{category} {title} {body}".lower()
    title_lower = title.lower()
    body_lower = body.lower()
    category_lower = category.lower()
    terms = _question_topic_terms(question)
    specific_terms = _question_specific_terms(question)
    if not terms:
        return 0

    score = 0
    specific_match_count = 0
    if category_key and category_key in routed_categories:
        score += 14
    for term in terms:
        in_title = term in title_lower
        in_body = term in body_lower
        in_category = term in category_lower
        if in_title:
            score += 4
        if in_body:
            score += 2
        if in_category:
            score += 5
        if term in specific_terms and (in_title or in_body):
            specific_match_count += 1

    if PRICING_HINT_RE.search(lower_question) and PRICING_HINT_RE.search(haystacks):
        score += 8
    if PRICING_HINT_RE.search(lower_question):
        score += 10 if PRICE_VALUE_RE.search(title) or PRICE_VALUE_RE.search(body) else -6
    if INSURANCE_HINT_RE.search(lower_question) and INSURANCE_HINT_RE.search(haystacks):
        score += 6
    if HOURS_HINT_RE.search(lower_question) and HOURS_HINT_RE.search(haystacks):
        score += 6
    if LOCATION_HINT_RE.search(lower_question) and LOCATION_HINT_RE.search(haystacks):
        score += 6
    if PARKING_HINT_RE.search(lower_question) and PARKING_HINT_RE.search(haystacks):
        score += 6
    if PAYMENT_HINT_RE.search(lower_question) and PAYMENT_HINT_RE.search(haystacks):
        score += 6
    if STAFF_HINT_RE.search(lower_question) and STAFF_HINT_RE.search(haystacks):
        score += 6
    if POLICY_HINT_RE.search(lower_question) and POLICY_HINT_RE.search(haystacks):
        score += 6
    if EMERGENCY_HINT_RE.search(lower_question) and EMERGENCY_HINT_RE.search(haystacks):
        score += 6
    if LOGISTICS_HINT_RE.search(lower_question) and LOGISTICS_HINT_RE.search(haystacks):
        score += 6

    detected_service = extract_reason_quick(question)
    if detected_service and detected_service.lower() in haystacks:
        score += 8
    if DETAIL_REQUEST_RE.search(lower_question):
        score += 2
    if routed_categories.intersection({"pricing", "services"}):
        if specific_terms and specific_match_count == 0 and not detected_service:
            score -= 16
        elif specific_match_count:
            score += specific_match_count * 4

    return score


def _rank_knowledge_articles(question: str, articles: Sequence[Dict[str, str]]) -> list[tuple[int, Dict[str, str]]]:
    ranked: list[tuple[int, Dict[str, str]]] = []
    for article in articles:
        title = str(article.get("title") or "")
        body = str(article.get("body") or "")
        category = str(article.get("category") or "")
        score = _knowledge_match_score(question, title=title, body=body, category=category)
        if score >= 4:
            ranked.append((score, article))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _best_knowledge_article(question: str, articles: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    ranked = _rank_knowledge_articles(question, articles)
    if not ranked:
        return None
    return ranked[0][1]


def _voice_answer_text(text: str, *, max_words: int = 48) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).rstrip(",;:") + "..."
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _select_knowledge_articles_for_answer(
    question: str,
    articles: Sequence[Dict[str, str]],
    *,
    limit: int = 3,
) -> list[Dict[str, str]]:
    ranked = _rank_knowledge_articles(question, articles)
    if not ranked:
        return []

    top_score = ranked[0][0]
    wants_details = bool(DETAIL_REQUEST_RE.search(question))
    min_score = max(4, top_score - (8 if wants_details else 4))

    # When a specific service is detected (e.g. "teeth whitening"), only return
    # articles whose title+body actually contain a service-specific term.
    # This prevents unrelated articles (e.g. "Root Canal", "Night Guards") from
    # being included just because a generic word like "teeth" appears in them.
    detected_service = extract_reason_quick(question)
    service_filter_terms = _service_specific_terms(detected_service) if detected_service else set()

    selected: list[Dict[str, str]] = []
    seen_bodies: set[str] = set()
    for score, article in ranked:
        if score < min_score:
            break

        if service_filter_terms:
            article_text = (
                f"{article.get('title', '')} {article.get('body', '')}"
            ).lower()
            if not any(term in article_text for term in service_filter_terms):
                continue

        body_key = " ".join(
            (
                str(article.get("title") or ""),
                str(article.get("body") or ""),
                str(article.get("category") or ""),
            )
        ).strip()
        if body_key in seen_bodies:
            continue
        selected.append(article)
        seen_bodies.add(body_key)
        if len(selected) >= limit:
            break

    return selected or [ranked[0][1]]


def _render_knowledge_article(question: str, article: Dict[str, str]) -> str:
    title = " ".join(str(article.get("title") or "").split()).strip()
    body = " ".join(str(article.get("body") or "").split()).strip()
    lower_question = question.lower()

    if STAFF_HINT_RE.search(lower_question) and re.search(r"\bname\b", lower_question) and title:
        details = body if body else title
        return _voice_answer_text(f"The doctor's name is {title}. {details}")

    detected_service = extract_reason_quick(question)
    specific_terms = _question_specific_terms(question)
    wants_pricing = PRICING_HINT_RE.search(lower_question) is not None
    sentences = [
        " ".join(sentence.split()).strip()
        for sentence in KNOWLEDGE_SENTENCE_SPLIT_RE.split(body)
        if " ".join(sentence.split()).strip()
    ]
    if body and len(sentences) > 1:
        service_terms = _service_specific_terms(detected_service)
        scored_sentences: list[tuple[int, int, str, bool]] = []
        for idx, sentence in enumerate(sentences):
            lower_sentence = sentence.lower()
            score = 0
            service_match = False
            sentence_service = extract_reason_quick(sentence)

            if detected_service and sentence_service and sentence_service.lower() == detected_service.lower():
                score += 18
                service_match = True
            if service_terms and any(term in lower_sentence for term in service_terms):
                score += 12
                service_match = True
            for term in specific_terms:
                if term in lower_sentence:
                    score += 4
            if wants_pricing and PRICE_VALUE_RE.search(sentence):
                score += 4
            if wants_pricing and PRICING_HINT_RE.search(lower_sentence):
                score += 2
            if score > 0:
                scored_sentences.append((score, idx, sentence, service_match))

        selected_sentences: list[str] = []
        if scored_sentences:
            service_specific = [item for item in scored_sentences if item[3]]
            if service_specific:
                _, best_idx, best_sentence, _ = max(
                    service_specific,
                    key=lambda item: (item[0], -item[1]),
                )
                selected_sentences.append(best_sentence)
                next_idx = best_idx + 1
                if next_idx < len(sentences):
                    next_sentence = sentences[next_idx]
                    next_service = extract_reason_quick(next_sentence)
                    next_conflicts = bool(
                        detected_service
                        and next_service
                        and next_service.lower() != detected_service.lower()
                    )
                    if not next_conflicts and PRICE_VALUE_RE.search(next_sentence):
                        selected_sentences.append(next_sentence)
            else:
                best_sentence = max(scored_sentences, key=lambda item: (item[0], -item[1]))[2]
                selected_sentences.append(best_sentence)

        if selected_sentences:
            combined = " ".join(selected_sentences)
            return _voice_answer_text(combined, max_words=60 if DETAIL_REQUEST_RE.search(lower_question) else 48)

    focused_excerpt = _service_focused_excerpt(question, body)
    if focused_excerpt:
        return _voice_answer_text(
            focused_excerpt,
            max_words=60 if DETAIL_REQUEST_RE.search(lower_question) else 48,
        )

    if body:
        return _voice_answer_text(body, max_words=60 if DETAIL_REQUEST_RE.search(lower_question) else 48)
    if title:
        return _voice_answer_text(title)
    return ""


def _service_focused_excerpt(question: str, body: str) -> Optional[str]:
    cleaned_body = " ".join((body or "").split()).strip()
    if not cleaned_body:
        return None

    detected_service = extract_reason_quick(question)
    if not detected_service:
        return None

    service_terms = sorted(_service_specific_terms(detected_service), key=len, reverse=True)
    if not service_terms:
        return None

    lower_body = cleaned_body.lower()
    service_positions = [
        match.start()
        for term in service_terms
        for match in re.finditer(re.escape(term), lower_body)
    ]
    if not service_positions:
        return None

    first_service_pos = min(service_positions)
    start = 0
    for match in re.finditer(r"[.!?;]\s+", cleaned_body):
        if match.end() <= first_service_pos:
            start = match.end()
        else:
            break

    conflict_pos: Optional[int] = None
    service_term_set = {term.lower() for term in service_terms}
    for term in SERVICE_BOUNDARY_TERMS:
        normalized_term = term.lower()
        if any(
            normalized_term == current
            or normalized_term in current
            or current in normalized_term
            for current in service_term_set
        ):
            continue
        for match in re.finditer(re.escape(term), lower_body):
            if match.start() <= first_service_pos:
                continue
            conflict_pos = match.start() if conflict_pos is None else min(conflict_pos, match.start())
            break

    end = len(cleaned_body)
    if conflict_pos is not None:
        previous_boundary_end: Optional[int] = None
        for match in re.finditer(r"[.!?;]\s+", cleaned_body):
            if match.end() <= conflict_pos:
                previous_boundary_end = match.end()
                continue
            break
        if previous_boundary_end is not None and previous_boundary_end > first_service_pos:
            end = previous_boundary_end - 1
        else:
            end = conflict_pos

    excerpt = cleaned_body[start:end].strip(" ,;:")
    if not excerpt:
        return None
    if PRICING_HINT_RE.search(question) and not PRICE_VALUE_RE.search(excerpt):
        return None
    return excerpt


def _compose_knowledge_answer(question: str, articles: Sequence[Dict[str, str]]) -> Optional[str]:
    selected = _select_knowledge_articles_for_answer(question, articles)
    if not selected:
        return None

    parts: list[str] = []
    for article in selected:
        rendered = _render_knowledge_article(question, article)
        if rendered and rendered not in parts:
            parts.append(rendered)

    if not parts:
        return None
    return " ".join(parts)


def _has_conflicting_service_mentions(text: str, target_service: Optional[str]) -> bool:
    normalized = " ".join((text or "").split()).strip().lower()
    if not normalized:
        return False

    target_terms = {term.lower() for term in _service_specific_terms(target_service)}
    if not target_terms and target_service:
        target_terms.add(" ".join(str(target_service).lower().split()))

    for term in SERVICE_BOUNDARY_TERMS:
        normalized_term = term.lower()
        if any(
            normalized_term == current
            or normalized_term in current
            or current in normalized_term
            for current in target_terms
        ):
            continue
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized):
            return True
    return False


def compose_clinic_info_answer(
    question: str,
    articles: Optional[Sequence[Dict[str, Any]]],
    *,
    fallback_service: Optional[str] = None,
) -> Optional[str]:
    normalized_articles = _normalize_knowledge_articles(articles)
    normalized = " ".join((question or "").split()).strip()
    contextual_question = _question_with_service_context(
        normalized,
        fallback_service,
    )
    if not _looks_like_clinic_info_question(
        contextual_question,
        knowledge_articles=normalized_articles,
    ):
        return None

    routed_categories = _question_knowledge_categories(contextual_question)
    ranked_articles = _rank_knowledge_articles(contextual_question, normalized_articles)
    detected_service = extract_reason_quick(contextual_question)
    if routed_categories.intersection({"pricing", "services"}) and not detected_service:
        specific_terms = _question_specific_terms(contextual_question)
        top_specific_match = max(
            (
                _specific_knowledge_match_count(
                    contextual_question,
                    title=str(article.get("title") or ""),
                    body=str(article.get("body") or ""),
                )
                for _, article in ranked_articles[:3]
            ),
            default=0,
        )
        if not specific_terms or top_specific_match == 0:
            return "I want to make sure I give you the right pricing. Which treatment would you like details for?"

    composed = _compose_knowledge_answer(contextual_question, normalized_articles)
    if composed:
        return composed

    service = detected_service or fallback_service
    service_phrase = service.lower() if isinstance(service, str) else "that service"
    lower = contextual_question.lower()

    if PRICING_HINT_RE.search(lower):
        return f"I don't have the exact pricing for {service_phrase} in my notes right now, but the office can confirm the current rate for you."
    if INSURANCE_HINT_RE.search(lower):
        return "I don't have the exact insurance details in my notes right now, but the office can confirm coverage for you."
    if HOURS_HINT_RE.search(lower):
        return "I don't have the exact office hours in my notes right now, but the office can confirm them for you."
    if LOCATION_HINT_RE.search(lower) or PARKING_HINT_RE.search(lower):
        return "I don't have that location detail in my notes right now, but the office can confirm it for you."
    return "I don't have that exact detail in my notes right now, but the office can confirm it for you."


def prune_clinic_response_for_tts(
    user_question: Optional[str],
    spoken_text: Optional[str],
    articles: Optional[Sequence[Dict[str, Any]]],
    *,
    fallback_service: Optional[str] = None,
) -> str:
    normalized_spoken = " ".join((spoken_text or "").split()).strip()
    normalized_question = " ".join((user_question or "").split()).strip()
    if not normalized_spoken or not normalized_question:
        return normalized_spoken

    contextual_question = _question_with_service_context(
        normalized_question,
        fallback_service,
    )
    routed_categories = _question_knowledge_categories(contextual_question)
    if not routed_categories.intersection({"pricing", "services"}):
        return normalized_spoken

    deterministic = compose_clinic_info_answer(
        contextual_question,
        articles,
        fallback_service=fallback_service,
    )
    if not deterministic:
        return normalized_spoken

    preserve_follow_up = "is there anything else i can help" in normalized_spoken.lower()
    sanitized = deterministic
    if preserve_follow_up and ANYTHING_ELSE_FOLLOW_UP_TEXT.lower() not in sanitized.lower():
        sanitized = f"{sanitized} {ANYTHING_ELSE_FOLLOW_UP_TEXT}"

    spoken_word_count = len(normalized_spoken.split())
    sanitized_word_count = len(sanitized.split())
    detected_service = extract_reason_quick(contextual_question) or fallback_service

    if _has_conflicting_service_mentions(normalized_spoken, detected_service):
        return sanitized
    if spoken_word_count > sanitized_word_count + 8:
        return sanitized
    return normalized_spoken


def _looks_like_clinic_info_question(
    question: Optional[str],
    *,
    knowledge_articles: Optional[Sequence[Dict[str, str]]] = None,
) -> bool:
    normalized = " ".join((question or "").split()).strip().lower()
    if not normalized:
        return False
    if _question_knowledge_categories(normalized):
        return True
    if APPOINTMENT_FLOW_HINT_RE.search(normalized):
        return False
    if QUESTION_LEAD_IN_RE.search(normalized) and knowledge_articles:
        return _best_knowledge_article(normalized, knowledge_articles) is not None
    return False


# ============================================================================
# AssistantTools class
# ============================================================================

class AssistantTools:
    """Tool functions for the dental AI agent."""

    def __init__(
        self,
        state: PatientState,
        clinic_info: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        schedule: Optional[Dict[str, Any]] = None,
        clinic_tz: Optional[str] = None,
        knowledge_articles: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        self.state = state
        self._clinic_info: Dict[str, Any] = clinic_info if clinic_info is not None else (_GLOBAL_CLINIC_INFO or {})
        self._settings: Dict[str, Any] = settings if settings is not None else (_GLOBAL_AGENT_SETTINGS or {})
        self._schedule: Dict[str, Any] = schedule if schedule is not None else (_GLOBAL_SCHEDULE or {})
        self._clinic_tz: str = clinic_tz or _GLOBAL_CLINIC_TZ
        self._knowledge_articles: list[Dict[str, str]] = _normalize_knowledge_articles(knowledge_articles)
        self._refresh_memory: Optional[Callable[[], None]] = None
        # Optional async callback for speaking a final response directly from a tool,
        # bypassing the LLM re-generation step. Set by agent.py after session is ready.
        # Signature: async def _direct_say_callback(text: str) -> None
        self._direct_say_callback: Optional[Callable] = None
        self._schedule_auto_disconnect: Optional[Callable] = None
        if self._clinic_info:
            logger.info(f"[TOOLS] Instance clinic context updated: {self._clinic_info.get('name')}, tz={self._clinic_tz}")

    def update_clinic_context(
        self,
        clinic_info: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
        schedule: Optional[Dict[str, Any]] = None,
        clinic_tz: Optional[str] = None,
        knowledge_articles: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Update clinic context after deferred DB load completes."""
        self._clinic_info = clinic_info or {}
        if settings is not None:
            self._settings = settings
        if schedule is not None:
            self._schedule = schedule
        if clinic_tz:
            self._clinic_tz = clinic_tz
        if knowledge_articles is not None:
            self._knowledge_articles = _normalize_knowledge_articles(knowledge_articles)
        logger.info(f"[TOOLS] Instance clinic context updated: {clinic_info.get('name')}, tz={self._clinic_tz}")

    def can_answer_clinic_question(self, question: Optional[str]) -> bool:
        contextual_question = _question_with_service_context(
            " ".join((question or "").split()).strip(),
            getattr(self.state, "reason", None),
        )
        return _looks_like_clinic_info_question(
            contextual_question,
            knowledge_articles=self._knowledge_articles,
        )

    def _compose_clinic_info_answer(self, question: str) -> Optional[str]:
        return compose_clinic_info_answer(
            question,
            self._knowledge_articles,
            fallback_service=getattr(self.state, "reason", None),
        )

    @llm.function_tool(
        description=(
            "Look up clinic FAQ, pricing, insurance, hours, location, parking, and service details "
            "from the clinic knowledge bank."
        )
    )
    async def search_clinic_info(self, question: str) -> str:
        answer = self._compose_clinic_info_answer(question)
        if answer:
            return answer
        return "I don't have that exact detail in my notes right now, but the office can confirm it for you."

    async def answer_clinic_question(
        self,
        question: str,
        *,
        include_follow_up: bool = False,
    ) -> Optional[str]:
        if not self.can_answer_clinic_question(question):
            return None

        answer = self._compose_clinic_info_answer(question)
        if not answer:
            answer = "I don't have that exact detail in my notes right now, but the office can confirm it for you."
        if not include_follow_up:
            return answer

        state = self.state
        state.anything_else_pending = True
        state.anything_else_asked = True
        state.user_declined_more_help = False
        state.final_goodbye_sent = False
        state.user_goodbye_detected = False
        state.closing_state = "anything_else_pending"
        _refresh_memory()
        return f"{answer} {ANYTHING_ELSE_FOLLOW_UP_TEXT}"

    @llm.function_tool(
        description=(
            "Save patient info. Handles: name, phone, email, reason, time_suggestion (natural language like 'March 10 at 3pm'). "
            "Checks availability automatically when time is given."
        )
    )
    async def update_patient_record(
        self,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        reason: Optional[str] = None,
        time_suggestion: Optional[str] = None,
    ) -> str:
        _t0 = time.perf_counter()
        state = self.state
        if not state:
            return "State not initialized."

        schedule = self._schedule or {}
        updates = []

        name = normalize_patient_name(_sanitize_tool_arg(name))
        phone = _sanitize_tool_arg(phone)
        email = _sanitize_tool_arg(email)
        reason = _sanitize_tool_arg(reason)
        time_suggestion = _sanitize_tool_arg(time_suggestion)

        spoken_name = normalize_patient_name(extract_name_quick(state.last_user_text or ""))

        if (
            name
            and state.full_name
            and name.strip().lower() != state.full_name.strip().lower()
            and (not spoken_name or spoken_name.strip().lower() != name.strip().lower())
        ):
            logger.warning(
                "[TOOL] Ignoring unsupported name overwrite: current=%s incoming=%s last_user_text=%r",
                state.full_name,
                name,
                state.last_user_text,
            )
            name = None

        if name and state.full_name and name.strip().lower() == state.full_name.strip().lower():
            name = None
        if reason and state.reason and reason.strip().lower() == state.reason.strip().lower():
            reason = None

        # === NAME ===
        if name:
            state.full_name = name.strip().title()
            updates.append(f"name={state.full_name}")
            logger.info(f"[TOOL] Name: {state.full_name}")

        # === PHONE ===
        if phone and not (state.phone_confirmed and state.phone_e164):
            clinic_region = (self._clinic_info or {}).get("default_phone_region", DEFAULT_PHONE_REGION)
            clean_phone, last4 = _normalize_phone_preserve_plus(phone, clinic_region)
            if clean_phone:
                state.phone_pending = str(clean_phone)
                state.phone_last4 = str(last4) if last4 else ""
                state.phone_confirmed = False
                state.phone_source = "user_spoken"
                updates.append(f"phone_pending=***{state.phone_last4}")
                logger.info(f"[TOOL] Phone pending: ***{state.phone_last4}")

        # === EMAIL ===
        if email:
            clean_email = email.replace(" ", "").lower()
            if "@" in clean_email and "." in clean_email:
                state.email = clean_email
                state.email_confirmed = False
                updates.append(f"email={state.email}")
                logger.info(f"[TOOL] Email: {state.email}")

        # === REASON ===
        if reason:
            state.reason = reason.strip()
            state.duration_minutes = get_duration_for_service(state.reason, schedule)
            updates.append(f"reason={state.reason} ({state.duration_minutes}m)")
            logger.info(f"[TOOL] Reason: {state.reason}, duration: {state.duration_minutes}m")

        # === TIME ===
        if time_suggestion:
            previous_dt_text = " ".join((state.dt_text or "").split()).strip() or None
            previous_dt_local = state.dt_local
            known_date_text = _date_hint_for_prompt(state, fallback_text=previous_dt_text)
            cleaned_suggestion = " ".join(time_suggestion.split()).strip()
            state.time_status = "validating"
            logger.info(f"[TOOL] Checking time: {time_suggestion}")

            try:
                # If we have a saved date and the user gives a time-only string, combine them
                time_only_words = ["am", "pm", "o'clock", "oclock"]
                month_words = [
                    "january","february","march","april","may","june","july",
                    "august","september","october","november","december",
                    "tomorrow","today","monday","tuesday","wednesday",
                    "thursday","friday","saturday","sunday",
                ]
                suggestion_lower = cleaned_suggestion.lower()
                has_date = any(m in suggestion_lower for m in month_words) or bool(re.search(r"\b\d{1,2}[/-]\d{1,2}\b", suggestion_lower))
                has_time = any(w in suggestion_lower for w in time_only_words) or bool(re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b", suggestion_lower, re.IGNORECASE))

                parse_input = cleaned_suggestion
                if not has_date and has_time and previous_dt_text and previous_dt_text != cleaned_suggestion:
                    # User gave time-only (e.g. "3 PM") — combine with saved date text
                    combined = f"{previous_dt_text} at {cleaned_suggestion}"
                    logger.info(f"[TOOL] Combined date+time: '{combined}'")
                    parse_input = combined

                result = parse_datetime_natural(parse_input, tz_hint=BOOKING_TZ)
                recent_context = state.recent_user_context() if hasattr(state, "recent_user_context") else ""
                parse_candidates = build_time_parse_candidates(
                    cleaned_suggestion,
                    recent_context=recent_context,
                    previous_text=known_date_text or previous_dt_text,
                )
                parse_candidates = parse_candidates or [parse_input]
                for candidate in parse_candidates:
                    if candidate == parse_input:
                        continue
                    attempt = parse_datetime_natural(candidate, tz_hint=BOOKING_TZ)
                    if (
                        attempt.get("datetime") is not None
                        or attempt.get("date_only")
                        or attempt.get("needs_clarification")
                        or attempt.get("clarification_type")
                    ):
                        parse_input = candidate
                        result = attempt
                        logger.info(f"[TOOL] Time parse override: '{time_suggestion}' -> '{parse_input}'")
                        break

                # Handle date-only result (no time was specified by user)
                if result.get("date_only") or result.get("clarification_type") == "time_missing":
                    parsed_date = result.get("parsed_date") or (result["datetime"].date() if result.get("datetime") else None)
                    if parsed_date:
                        state.dt_text = parse_input
                        state.time_status = "pending"
                        state.time_error = None
                        state.slot_available = False
                        day_spoken = parsed_date.strftime("%A, %B %d")
                        _refresh_memory()
                        return f"Got it, {day_spoken}. What time works best for you?"
                    state.dt_text = known_date_text or previous_dt_text
                    state.time_status = "pending"
                    state.time_error = "time_missing"
                    state.slot_available = False
                    _refresh_memory()
                    return _time_reask_text(state, date_hint=known_date_text)

                if result.get("needs_clarification"):
                    state.dt_text = known_date_text or previous_dt_text
                    state.time_status = "pending"
                    state.time_error = str(result.get("clarification_type") or "clarification_needed")
                    state.slot_available = False
                    _refresh_memory()
                    return result.get("message", "Could you specify the day?")

                parsed = result.get("datetime")

                if not parsed:
                    state.dt_text = (
                        known_date_text
                        or (cleaned_suggestion if has_date_reference(cleaned_suggestion) else previous_dt_text)
                    )
                    state.time_status = "pending"
                    state.time_error = "time_parse_failed"
                    state.slot_available = False
                    state.dt_local = previous_dt_local
                    _refresh_memory()
                    if known_date_text:
                        return _time_reask_text(state, date_hint=known_date_text)
                    return "I didn't catch that. What day and time would you like?"

                booking_tz = ZoneInfo(BOOKING_TZ)
                parsed = (
                    parsed.astimezone(booking_tz)
                    if parsed.tzinfo is not None
                    else parsed.replace(tzinfo=booking_tz)
                )
                state.dt_text = parse_input
                logger.info(f"[TOOL] Parsed '{time_suggestion}' → {parsed.isoformat()}")

                time_spoken = parsed.strftime("%I:%M %p").lstrip("0")
                day_spoken = parsed.strftime("%A")

                is_valid, error_msg = is_within_working_hours(parsed, schedule, state.duration_minutes)

                if is_valid:
                    clinic_id = str((self._clinic_info or {}).get("id") or "")
                    if clinic_id:
                        slot_end = parsed + timedelta(minutes=state.duration_minutes + APPOINTMENT_BUFFER_MINUTES)
                        clinic_info = self._clinic_info or {}
                        slot_free = await is_slot_free_supabase(
                            clinic_id,
                            parsed,
                            slot_end,
                            clinic_info=clinic_info,
                        )

                        if not slot_free:
                            state.time_status = "invalid"
                            state.time_error = "That slot is already taken"
                            state.add_rejected_slot(parsed, reason="slot_taken")
                            state.dt_local = None
                            state.slot_available = False

                            alternatives = await suggest_slots_around(
                                clinic_id=clinic_id,
                                requested_start_dt=parsed,
                                duration_minutes=state.duration_minutes,
                                schedule=schedule,
                                tz_str=BOOKING_TZ,
                                count=3,
                                window_hours=4,
                                step_min=15,
                            )
                            valid_alts = [a for a in alternatives if not state.is_slot_rejected(a)]

                            if valid_alts:
                                alt_strs = []
                                for alt in valid_alts:
                                    t = alt.strftime("%I:%M %p").lstrip("0")
                                    alt_strs.append(t if alt.date() == parsed.date() else f"{alt.strftime('%A')} at {t}")
                                if len(alt_strs) == 1:
                                    return f"Sorry, {time_spoken} is booked. The closest I have is {alt_strs[0]}. Would that work?"
                                elif len(alt_strs) == 2:
                                    return f"Sorry, {time_spoken} is booked. I can do {alt_strs[0]} or {alt_strs[1]}."
                                else:
                                    return f"Sorry, {time_spoken} is booked. I can do {alt_strs[0]}, {alt_strs[1]}, or {alt_strs[2]}."
                            return f"Sorry, {time_spoken} on {day_spoken} is booked and I don't see nearby openings. Try another day?"

                    # Slot is free (or no clinic_id to check)
                    state.dt_local = parsed
                    state.time_status = "valid"
                    state.time_error = None
                    state.slot_available = True
                    updates.append(f"time={parsed.strftime('%A, %B %d at %I:%M %p')}")
                    logger.info(f"[TOOL] Time confirmed available: {parsed.isoformat()}")

                    if state.full_name and state.dt_local:
                        state.contact_phase_started = True

                    # Caller ID flow
                    if contact_phase_allowed(state) and not state.phone_confirmed and not state.caller_id_checked:
                        phone_candidate = ensure_caller_phone_pending(state)
                        if phone_candidate:
                            state.pending_confirm = "phone"
                            state.pending_confirm_field = "phone"
                            state.caller_id_checked = True
                            _refresh_memory()
                            phone_prompt = _phone_confirmation_question(state, phone_candidate)
                            return f"Perfect. {day_spoken} at {time_spoken} is open. {phone_prompt}"

                    _refresh_memory()
                    return f"Got it — {day_spoken} at {time_spoken} is available. Continue gathering remaining info."

                else:
                    state.time_status = "invalid"
                    state.time_error = error_msg
                    state.dt_local = None
                    clinic_id = str((self._clinic_info or {}).get("id") or "")
                    start_date = parsed.date() if parsed else datetime.now(ZoneInfo(BOOKING_TZ)).date()
                    alternatives = await get_next_available_slots(
                        clinic_id=clinic_id,
                        schedule=schedule,
                        tz_str=BOOKING_TZ,
                        duration_minutes=state.duration_minutes,
                        num_slots=2,
                        days_ahead=7,
                        start_from_date=start_date,
                    )
                    if alternatives:
                        times = [t.strftime("%I:%M %p").lstrip("0") for t in alternatives]
                        return (f"{error_msg} I have {times[0]}"
                                + (f" or {times[1]}" if len(times) > 1 else "")
                                + ". Would you like one of those?")
                    return f"{error_msg} Would you like to try another time?"

            except Exception as e:
                logger.error(f"[TOOL] Time validation error: {e!r}")
                state.dt_text = (
                    known_date_text
                    or (cleaned_suggestion if has_date_reference(cleaned_suggestion) else previous_dt_text)
                )
                state.time_status = "pending"
                state.time_error = "schedule_unavailable"
                state.dt_local = previous_dt_local
                state.slot_available = False
                _refresh_memory()
                if known_date_text:
                    return _time_reask_text(state, date_hint=known_date_text)
                return "I'm having trouble checking the schedule. Could you try a different day or time?"

        # Caller ID flow after other updates
        if state.full_name and state.dt_local:
            state.contact_phase_started = True

        if contact_phase_allowed(state) and not state.phone_confirmed and not state.caller_id_checked:
            phone_candidate = ensure_caller_phone_pending(state)
            if phone_candidate and state.pending_confirm != "phone":
                state.pending_confirm = "phone"
                state.pending_confirm_field = "phone"
                state.caller_id_checked = True
                _refresh_memory()
                return _phone_confirmation_question(state, phone_candidate)

        _refresh_memory()
        _ms = (time.perf_counter() - _t0) * 1000
        logger.info(f"[PERF] update_patient_record: {_ms:.0f}ms")

        return "Noted."

    @llm.function_tool(description="Confirm or reject caller's phone number. confirmed=True saves it, confirmed=False clears and re-asks.")
    async def confirm_phone(self, confirmed: bool, new_phone: Optional[str] = None, phone_number: Optional[str] = None) -> str:
        state = self.state
        state.pending_confirm_field = None
        state.pending_confirm = None

        if state.phone_confirmed and state.phone_e164 and confirmed and not new_phone and not phone_number:
            if state.full_name and state.dt_local and state.reason:
                return "Phone saved. All info complete. Book now."
            return "Phone saved."

        new_phone = _sanitize_tool_arg(phone_number) or _sanitize_tool_arg(new_phone)

        if new_phone:
            clinic_region = (self._clinic_info or {}).get("default_phone_region", DEFAULT_PHONE_REGION)
            clean_phone, last4 = _normalize_phone_preserve_plus(new_phone, clinic_region)
            if clean_phone:
                existing_phone = state.phone_pending or state.detected_phone or state.phone_e164
                if (
                    existing_phone
                    and str(clean_phone) == str(existing_phone)
                    and not looks_like_phone_input(getattr(state, "last_user_text", None))
                ):
                    logger.warning("[TOOL] Ignoring synthetic phone update without caller phone input")
                    _refresh_memory()
                    return "Wait for the caller to answer whether to use the caller ID number."
                state.phone_pending = str(clean_phone)
                state.phone_last4 = str(last4) if last4 else ""
                state.phone_confirmed = False
                state.phone_source = "user_spoken"
                state.using_caller_number = False
                state.confirmed_contact_number_source = None
                state.pending_confirm = "phone"
                state.pending_confirm_field = "phone"
                logger.info(f"[TOOL] Phone updated: ***{state.phone_last4}")
                _refresh_memory()
                return _phone_confirmation_question(state, str(clean_phone))
            return f"Couldn't parse '{new_phone}'. Could you repeat the number?"

        if confirmed:
            phone_candidate = ensure_caller_phone_pending(state)
            if not phone_candidate:
                return "No phone number to confirm. Ask for the number first."
            state.phone_e164 = str(phone_candidate)
            state.phone_confirmed = True
            using_caller_number = (state.phone_source or "").lower() == "sip"
            state.caller_id_accepted = using_caller_number
            state.using_caller_number = using_caller_number
            state.confirmed_contact_number_source = "caller_id" if using_caller_number else "user_provided"
            state.contact_phase_started = True
            state.pending_confirm = None if state.pending_confirm == "phone" else state.pending_confirm
            state.pending_confirm_field = None if state.pending_confirm_field == "phone" else state.pending_confirm_field
            logger.info(f"[TOOL] Phone confirmed: {state.phone_e164}")
            _refresh_memory()
            if state.full_name and state.dt_local and state.reason:
                return "Phone saved. All info complete. Book now."
            return "Phone saved."
        else:
            old = state.phone_pending or state.detected_phone or state.phone_e164
            state.phone_pending = None
            state.detected_phone = None
            state.phone_e164 = None
            state.phone_last4 = None
            state.phone_confirmed = False
            state.phone_source = None
            state.using_caller_number = False
            state.confirmed_contact_number_source = None
            state.caller_id_accepted = False
            state.pending_confirm = None if state.pending_confirm == "phone" else state.pending_confirm
            state.pending_confirm_field = None if state.pending_confirm_field == "phone" else state.pending_confirm_field
            logger.info(f"[TOOL] Phone rejected (was {old})")
            _refresh_memory()
            return "Phone cleared. Ask: 'What number should I use instead?'"

    @llm.function_tool(description="Save whether appointment confirmation should be sent on WhatsApp or SMS.")
    async def set_delivery_preference(self, channel: str) -> str:
        state = self.state
        if not state:
            return "State not initialized."

        try:
            acknowledgement = _apply_delivery_preference(state, channel)
        except ValueError:
            return "Please choose either WhatsApp or SMS."

        if state.appointment_booked:
            state.anything_else_pending = True
            state.anything_else_asked = True
            state.user_declined_more_help = False
            state.final_goodbye_sent = False
            state.user_goodbye_detected = False
            state.closing_state = "anything_else_pending"
            _refresh_memory()
            return f"{acknowledgement} Is there anything else I can help you with today?"

        state.closing_state = "open"
        _refresh_memory()
        return acknowledgement

    @llm.function_tool(description="Confirm or reject patient's email address.")
    async def confirm_email(self, confirmed: bool, email_address: Optional[str] = None) -> str:
        state = self.state
        if not state:
            return "State not initialized."

        email_address = _sanitize_tool_arg(email_address)
        if email_address:
            state.email = email_address.strip().lower()
            logger.info(f"[TOOL] Email saved: {state.email}")

        if state.email_confirmed:
            return "Email already confirmed."

        if not contact_phase_allowed(state):
            return "Confirm appointment time first."

        if confirmed:
            state.email_confirmed = True
            state.pending_confirm = None if state.pending_confirm == "email" else state.pending_confirm
            state.pending_confirm_field = None if state.pending_confirm_field == "email" else state.pending_confirm_field
            logger.info("[TOOL] Email confirmed")
            _refresh_memory()
            return "Email saved."
        else:
            state.email = None
            state.email_confirmed = False
            logger.info("[TOOL] Email rejected")
            _refresh_memory()
            return "Email cleared. Ask for the correct email."

    @llm.function_tool(
        description=(
            "Find open time slots. ONLY use when patient asks 'what times are available' or 'anytime works'. "
            "Do NOT use when patient gives a specific date/time — use update_patient_record(time_suggestion=...) instead, it checks availability automatically."
        )
    )
    async def get_available_slots_v2(
        self,
        after_datetime: Optional[str] = None,
        preferred_day: Optional[str] = None,
        num_slots: int = 3,
    ) -> str:
        _t0 = time.perf_counter()
        state = self.state
        clinic_info = self._clinic_info
        schedule = self._schedule or {}

        if not clinic_info:
            return "I'm having trouble accessing the schedule right now."

        duration = state.duration_minutes if state else 60
        tz = ZoneInfo(BOOKING_TZ)
        now = datetime.now(tz)

        after_datetime = _sanitize_tool_arg(after_datetime)
        preferred_day = _sanitize_tool_arg(preferred_day)

        search_start = now
        if after_datetime:
            try:
                result = parse_datetime_natural(after_datetime, tz_hint=BOOKING_TZ)
                if result.get("success") and result.get("datetime"):
                    search_start = result["datetime"]
            except Exception as e:
                logger.warning(f"[TOOL] Could not parse after_datetime '{after_datetime}': {e}")

        target_weekday = None
        if preferred_day:
            day_map = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6,
                "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
            }
            day_lower = preferred_day.lower().strip()
            if day_lower == "today":
                target_weekday = now.weekday()
            elif day_lower == "tomorrow":
                target_weekday = (now + timedelta(days=1)).weekday()
            elif day_lower in day_map:
                target_weekday = day_map[day_lower]

        try:
            slot_step = schedule.get("slot_step_minutes", 30)
            minutes_to_add = slot_step - (search_start.minute % slot_step)
            if minutes_to_add == slot_step:
                minutes_to_add = 0
            current = search_start.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)

            if target_weekday is not None and current.weekday() != target_weekday:
                days_until = (target_weekday - current.weekday()) % 7
                if days_until == 0:
                    days_until = 7
                current = datetime.combine(
                    current.date() + timedelta(days=days_until),
                    datetime.min.time(), tzinfo=tz
                ).replace(hour=9, minute=0)

            end_search = max(now + timedelta(days=14), search_start + timedelta(days=14))

            existing_appointments = []
            try:
                result = await asyncio.to_thread(
                    lambda: supabase.table("appointments")
                    .select("start_time, end_time")
                    .eq("clinic_id", clinic_info["id"])
                    .gte("start_time", now.isoformat())
                    .lte("start_time", end_search.isoformat())
                    .in_("status", BOOKED_STATUSES)
                    .execute()
                )
                for appt in (result.data or []):
                    try:
                        appt_dict = cast(Dict[str, Any], appt)
                        appt_start = datetime.fromisoformat(str(appt_dict["start_time"]).replace("Z", "+00:00"))
                        appt_end = datetime.fromisoformat(str(appt_dict["end_time"]).replace("Z", "+00:00"))
                        existing_appointments.append((appt_start, appt_end))
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[SLOTS_V2] Failed to fetch appointments: {e}")

            available_slots = []
            lunch_skipped = False

            while current < end_search and len(available_slots) < num_slots:
                if target_weekday is not None and current.weekday() != target_weekday:
                    days_until = (target_weekday - current.weekday()) % 7
                    if days_until == 0:
                        days_until = 7
                    current = datetime.combine(
                        current.date() + timedelta(days=days_until),
                        datetime.min.time(), tzinfo=tz
                    ).replace(hour=9, minute=0)
                    if current >= end_search:
                        break

                is_valid, error_msg = is_within_working_hours(current, schedule, duration)
                if not is_valid and "lunch" in error_msg.lower():
                    lunch_skipped = True

                if is_valid:
                    slot_end = current + timedelta(minutes=duration + APPOINTMENT_BUFFER_MINUTES)
                    is_free = True
                    for appt_start, appt_end in existing_appointments:
                        if current < (appt_end + timedelta(minutes=APPOINTMENT_BUFFER_MINUTES)) and slot_end > appt_start:
                            is_free = False
                            break
                    if is_free:
                        available_slots.append(current)

                current += timedelta(minutes=slot_step)

                dow_key = WEEK_KEYS[current.weekday()]
                intervals = schedule.get("working_hours", {}).get(dow_key, [])
                if intervals:
                    last_interval = intervals[-1]
                    try:
                        eh, em = map(int, last_interval["end"].split(":"))
                        day_end = current.replace(hour=eh, minute=em)
                        if current >= day_end:
                            next_day = current.date() + timedelta(days=1)
                            current = datetime.combine(next_day, datetime.min.time(), tzinfo=tz).replace(hour=9, minute=0)
                    except Exception:
                        pass

            if state:
                available_slots = [s for s in available_slots if not state.is_slot_rejected(s)]

            if not available_slots:
                constraint = f" after {after_datetime}" if after_datetime else ""
                if preferred_day:
                    constraint += f" on {preferred_day}"
                return f"I don't see any openings{constraint} in the next week. Try another day?"

            slot_strings = []
            for slot in available_slots:
                t = slot.strftime("%I:%M %p").lstrip("0")
                today = now.date()
                if slot.date() == today:
                    slot_strings.append(f"today at {t}")
                elif slot.date() == today + timedelta(days=1):
                    slot_strings.append(f"tomorrow at {t}")
                else:
                    slot_strings.append(f"{slot.strftime('%A')} at {t}")

            prefix = "Okay, skipping the lunch hour. " if lunch_skipped else "I found some times. "
            if len(slot_strings) == 1:
                return f"{prefix}The next available is {slot_strings[0]}."
            elif len(slot_strings) == 2:
                return f"{prefix}I have {slot_strings[0]} or {slot_strings[1]}."
            else:
                return f"{prefix}I have {slot_strings[0]}, {slot_strings[1]}, or {slot_strings[2]}. Which works best?"

        except Exception as e:
            logger.error(f"[TOOL] get_available_slots_v2 error: {e}")
            return "I'm having trouble with the schedule. Let me try that again."
        finally:
            _ms = (time.perf_counter() - _t0) * 1000
            logger.info(f"[PERF] get_available_slots_v2: {_ms:.0f}ms")

    @llm.function_tool(description="Book the appointment after all info is confirmed by the patient.")
    async def confirm_and_book_appointment(self) -> str:
        _t0 = time.perf_counter()
        state = self.state
        clinic_info = self._clinic_info

        if not state or not clinic_info:
            return "Sorry, I'm missing clinic details. Could you call back in a moment?"

        if getattr(state, "appointment_booked", False):
            dt = state.dt_local
            if dt:
                day = dt.strftime("%A, %B %d")
                time_str = dt.strftime("%I:%M %p").lstrip("0")
                return f"You're already booked for {state.reason or 'your appointment'} on {day} at {time_str}."
            return "You're already booked."

        if not state.full_name or not state.dt_local or not state.phone_e164 or not state.phone_confirmed:
            missing = []
            if not state.full_name: missing.append("name")
            if not state.dt_local: missing.append("appointment time")
            if not state.phone_e164 or not state.phone_confirmed: missing.append("confirmed phone number")
            return f"I still need: {', '.join(missing)}. Let me get those first."

        booking_tz = ZoneInfo(BOOKING_TZ)
        state.dt_local = (
            state.dt_local.astimezone(booking_tz)
            if state.dt_local.tzinfo is not None
            else state.dt_local.replace(tzinfo=booking_tz)
        )

        # --- Booking-in-progress mutex: prevents duplicate inserts from concurrent fast-lane tasks ---
        if getattr(state, "booking_in_progress", False):
            logger.info("[BOOK] Booking already in progress — ignoring duplicate call (silent)")
            raise llm.StopResponse()

        state.booking_in_progress = True
        logger.info(f"[BOOK] Starting Supabase insert for {state.full_name}")
        appt_id = None
        try:
            appt_id = await book_to_supabase(clinic_info, patient_state=state, calendar_event_id=None)
        except Exception as e:
            logger.error(f"[BOOK] Supabase failed: {e!r}")
        finally:
            state.booking_in_progress = False

        if not appt_id:
            return "I'm having trouble saving the appointment. Could you try again?"

        state.appointment_booked = True
        state.booking_confirmed = True
        state.appointment_id = appt_id
        state.user_declined_more_help = False
        state.final_goodbye_sent = False
        state.user_goodbye_detected = False

        _ms = (time.perf_counter() - _t0) * 1000
        logger.info(f"[PERF] confirm_and_book: {_ms:.0f}ms, appt_id={appt_id}")
        booking_sentence = _booking_sentence(state)
        if state.delivery_channel not in {"whatsapp", "sms"} and state.prefers_sms:
            state.delivery_channel = "sms"

        if state.delivery_channel in {"whatsapp", "sms"}:
            target = _contact_reference(state)
            if state.delivery_channel == "sms":
                delivery_part = f" I'll send your confirmation by SMS to {target}."
            else:
                delivery_part = f" I'll send your confirmation on WhatsApp to {target}."
            state.delivery_preference_pending = False
            state.delivery_preference_asked = True
            state.anything_else_pending = True
            state.anything_else_asked = True
            state.closing_state = "anything_else_pending"
            _refresh_memory()
            return f"{booking_sentence}{delivery_part} Is there anything else I can help you with today?"

        state.delivery_preference_pending = True
        state.delivery_preference_asked = True
        state.delivery_ask_count = 0
        state.anything_else_pending = False
        state.anything_else_asked = False
        state.closing_state = "delivery_pending"
        _refresh_memory()
        return f"{booking_sentence} {_delivery_question_text(state)}"

    @llm.function_tool(description="Find existing appointment for cancel/reschedule. Call silently when user mentions cancelling or rescheduling.")
    async def find_existing_appointment(self) -> str:
        state = self.state
        clinic_info = self._clinic_info

        if not clinic_info:
            return "I'm having trouble accessing the system right now."

        phone_to_search = state.phone_e164 or state.phone_pending or state.detected_phone
        if isinstance(phone_to_search, tuple):
            phone_to_search = phone_to_search[0] if phone_to_search else None

        if not phone_to_search:
            return "I don't have a phone number to search with. What number did you use when booking?"

        logger.info(f"[APPT_LOOKUP] Searching for ***{phone_to_search[-4:]}")

        appointment = await find_appointment_by_phone(
            clinic_id=clinic_info["id"],
            phone_number=phone_to_search,
            tz_str=BOOKING_TZ
        )

        if appointment:
            state.found_appointment_id = appointment["id"]
            state.found_appointment_details = appointment
            start_time = appointment["start_time"]
            day = start_time.strftime("%A, %B %d")
            time_str = start_time.strftime("%I:%M %p").lstrip("0")
            logger.info(f"[APPT_LOOKUP] Found id={appointment['id']}")
            return f"I found your {appointment['reason']} appointment on {day} at {time_str}. Is this the one you'd like to modify?"
        else:
            logger.info(f"[APPT_LOOKUP] No appointment found for ***{phone_to_search[-4:]}")
            return "I don't see an upcoming appointment with that number. What number did you use when booking?"

    @llm.function_tool(description="Cancel found appointment after explicit user confirmation.")
    async def cancel_appointment_tool(self, confirmed: bool = False) -> str:
        state = self.state
        appointment_id = getattr(state, "found_appointment_id", None)
        appointment = getattr(state, "found_appointment_details", None)

        if not appointment_id or not appointment:
            return "I need to find your appointment first. Let me search using your phone number."

        if not confirmed:
            start_time = appointment["start_time"]
            day = start_time.strftime("%A, %B %d")
            time_str = start_time.strftime("%I:%M %p").lstrip("0")
            return f"Just to confirm — cancel your {appointment['reason']} on {day} at {time_str}?"

        logger.info(f"[CANCEL] Cancelling appointment id={appointment_id}")
        success = await cancel_appointment(appointment_id=appointment_id, reason="user_requested")

        if success:
            start_time = appointment["start_time"]
            day = start_time.strftime("%A, %B %d")
            time_str = start_time.strftime("%I:%M %p").lstrip("0")
            state.found_appointment_id = None
            state.found_appointment_details = None
            logger.info(f"[CANCEL] Cancelled id={appointment_id}")
            return f"Done — your {appointment['reason']} on {day} at {time_str} has been cancelled. Anything else?"
        else:
            return "I'm having trouble cancelling that appointment. Would you like to speak with the office?"

    @llm.function_tool(description="Reschedule found appointment to a new time after explicit user confirmation of both old and new times.")
    async def reschedule_appointment_tool(self, new_time: Optional[str] = None, confirmed: bool = False) -> str:
        state = self.state
        appointment_id = getattr(state, "found_appointment_id", None)
        appointment = getattr(state, "found_appointment_details", None)

        if not appointment_id or not appointment:
            return "I need to find your appointment first."

        if not new_time:
            return "Do you have a specific time in mind, or would you like me to suggest some options?"

        new_time = _sanitize_tool_arg(new_time)
        if not new_time:
            return "What time would work better for you?"

        try:
            parsed_result = parse_datetime_natural(new_time, tz_hint=BOOKING_TZ)
            parsed_new_time = parsed_result.get("datetime") if isinstance(parsed_result, dict) else parsed_result

            if not parsed_new_time:
                return f"I couldn't understand '{new_time}'. Could you try again?"

            booking_tz = ZoneInfo(BOOKING_TZ)
            parsed_new_time = (
                parsed_new_time.astimezone(booking_tz)
                if parsed_new_time.tzinfo is not None
                else parsed_new_time.replace(tzinfo=booking_tz)
            )

            clinic_info = self._clinic_info
            if not clinic_info:
                return "I'm having trouble accessing the system."

            schedule = self._schedule or {}
            duration = appointment.get("duration_minutes", 60)

            is_valid, error_msg = is_within_working_hours(parsed_new_time, schedule, duration)
            if not is_valid:
                return f"{error_msg} Would you like me to suggest available times?"

            end_time = parsed_new_time + timedelta(minutes=duration + APPOINTMENT_BUFFER_MINUTES)
            slot_free = await is_slot_free_supabase(
                clinic_id=clinic_info["id"],
                start_dt=parsed_new_time,
                end_dt=end_time,
                clinic_info=clinic_info
            )

            if not slot_free:
                alternatives = await suggest_slots_around(
                    clinic_id=clinic_info["id"],
                    requested_start_dt=parsed_new_time,
                    duration_minutes=duration,
                    schedule=schedule,
                    tz_str=BOOKING_TZ,
                    count=3,
                    window_hours=4,
                    step_min=15,
                )
                if alternatives:
                    alt_strs = []
                    for alt in alternatives:
                        t = alt.strftime("%I:%M %p").lstrip("0")
                        alt_strs.append(t if alt.date() == parsed_new_time.date() else f"{alt.strftime('%A')} at {t}")
                    if len(alt_strs) == 1:
                        return f"That slot is taken. The closest I have is {alt_strs[0]}. Would that work?"
                    elif len(alt_strs) == 2:
                        return f"That slot is booked. I can do {alt_strs[0]} or {alt_strs[1]}."
                    else:
                        return f"That time is taken. I have {alt_strs[0]}, {alt_strs[1]}, or {alt_strs[2]}."
                return "That slot is taken and I don't see nearby openings. Try a different day?"

            old_start = appointment["start_time"]
            old_day = old_start.strftime("%A, %B %d")
            old_time_str = old_start.strftime("%I:%M %p").lstrip("0")
            new_day = parsed_new_time.strftime("%A, %B %d")
            new_time_str = parsed_new_time.strftime("%I:%M %p").lstrip("0")

            if not confirmed:
                return (f"Just to confirm — move your {appointment['reason']} from {old_day} at {old_time_str} "
                        f"to {new_day} at {new_time_str}?")

            end_time = parsed_new_time + timedelta(minutes=duration)
            success = await reschedule_appointment(
                appointment_id=appointment_id,
                new_start_time=parsed_new_time,
                new_end_time=end_time,
            )

            if success:
                state.found_appointment_id = None
                state.found_appointment_details = None
                logger.info(f"[RESCHEDULE] Rescheduled id={appointment_id}")
                return f"All set — moved to {new_day} at {new_time_str}. Anything else?"
            else:
                return "I'm having trouble rescheduling. Would you like to speak with the office?"

        except Exception as e:
            logger.error(f"[RESCHEDULE] Error: {e}")
            return "I'm having trouble with that. Would you like to speak with the office directly?"

    @llm.function_tool(description="End the call when the user says goodbye or conversation is complete.")
    async def end_conversation(self) -> str:
        state = self.state
        if state:
            if state.booking_confirmed and state.delivery_preference_pending:
                return "Before ending the call, ask whether they'd like the confirmation on WhatsApp or by SMS."
            if state.booking_confirmed and not state.user_declined_more_help and not state.anything_else_pending:
                return "Before ending the call, ask if there is anything else you can help with today."

            state.call_ended = True
            state.final_goodbye_sent = True
            state.closing_state = "final_goodbye_sent"

            if state.booking_confirmed:
                logger.info("[CALL_END] Call ending after successful booking")
            else:
                logger.info("[CALL_END] Call ending at user request")

            if self._schedule_auto_disconnect:
                try:
                    self._schedule_auto_disconnect(None)
                except Exception as e:
                    logger.warning(f"[CALL_END] Failed to schedule disconnect: {e}")

        return "Wonderful. You're all set — we'll see you then. Have a great day."
