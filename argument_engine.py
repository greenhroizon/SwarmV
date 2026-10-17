"""Deterministic argument mapping for bounded Swarm Signal evidence.

The live provider is useful for collecting role reports and a synthesis, but it
is not an authority for graph structure or scores.  This module is the local
analysis boundary: it accepts only a bounded question, evidence packet, role
reports, and a few safe synthesis fields, then constructs the graph and scores
from those inputs.  Provider supplied graph fields (for example ``connections``
or ``impact_nodes``) are intentionally ignored.

There are no third-party dependencies and no network or model calls here.  The
same inputs produce the same output, which makes the argument map suitable for
offline tests and for clients that need a stable response shape.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


MAX_ARGUMENT_REPORTS = 12
MAX_ARGUMENT_EVIDENCE = 8
MAX_ARGUMENT_NODES = 64
MAX_ARGUMENT_EDGES = 192
MAX_ARGUMENT_TEXT = 1_200
MAX_ARGUMENT_LABEL = 300
MAX_ARGUMENT_SOURCE_ID = 80

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_SUPPORT_STANCES = {
    "support",
    "supportive",
    "positive",
    "for",
    "yes",
    "likely",
    "pro",
}
_OPPOSITION_STANCES = {
    "oppose",
    "opposed",
    "opposition",
    "skeptical",
    "against",
    "negative",
    "no",
    "unlikely",
    "con",
}


def _safe_text(value: Any, maximum: int = MAX_ARGUMENT_TEXT) -> str:
    """Return bounded, printable text without exposing control characters."""

    if not isinstance(value, str):
        return ""
    cleaned = _CONTROL_RE.sub(" ", value)
    cleaned = cleaned.encode("utf-8", errors="replace").decode("utf-8")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned[:maximum].strip()


def _safe_id(value: Any, fallback: str, maximum: int = MAX_ARGUMENT_SOURCE_ID) -> str:
    text = _safe_text(value, maximum)
    return text or fallback


def _bounded_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number):
        return default
    return max(0.0, min(1.0, number))


def _rounded(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


def _normalize_stance(value: Any, *, text: str = "") -> str:
    candidate = _safe_text(value, 40).casefold().replace("_", "-")
    if candidate in _SUPPORT_STANCES:
        return "supportive"
    if candidate in _OPPOSITION_STANCES:
        return "opposed"
    if candidate in {"mixed", "neutral", "uncertain", "balanced", "unknown"}:
        return "mixed"

    # Legacy/free-text reports do not have a stance field.  These cues are
    # deliberately small and transparent; they are only a fallback label, not
    # a claim that the text is evidence.
    tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    positive = len(tokens & {"support", "supports", "likely", "ready", "success", "approved"})
    negative = len(tokens & {"oppose", "opposes", "risk", "unlikely", "blocked", "fail", "failed"})
    if positive > negative:
        return "supportive"
    if negative > positive:
        return "opposed"
    return "mixed"


def _bounded_strings(value: Any, *, maximum_items: int, maximum_text: int = MAX_ARGUMENT_TEXT) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        return []
    result: list[str] = []
    for item in value[:maximum_items]:
        text = _safe_text(item, maximum_text)
        if text:
            result.append(text)
    return result


def _extract_evidence_items(evidence_items: Any) -> list[Any]:
    if isinstance(evidence_items, Mapping):
        for key in ("sources", "evidence", "evidence_items", "items"):
            candidate = evidence_items.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                return list(candidate)
        return []
    if isinstance(evidence_items, Sequence) and not isinstance(evidence_items, (str, bytes, bytearray)):
        return list(evidence_items)
    return []


def _normalize_evidence(evidence_items: Any) -> list[dict[str, Any]]:
    """Copy only the evidence fields used by the local engine."""

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(_extract_evidence_items(evidence_items)[:MAX_ARGUMENT_EVIDENCE]):
        if not isinstance(raw, Mapping):
            continue
        source_id = _safe_id(raw.get("id"), f"source-{index + 1:02d}")
        key = source_id.casefold()
        if key in seen_ids:
            continue
        seen_ids.add(key)
        status_value = raw.get("status", "ok")
        status = _safe_text(status_value, 24).casefold() if not isinstance(status_value, int) else (
            "ok" if status_value == 200 else "error"
        )
        title = _safe_text(raw.get("title", ""), MAX_ARGUMENT_LABEL)
        excerpt = _safe_text(
            raw.get("excerpt")
            or raw.get("summary")
            or raw.get("readable_text")
            or raw.get("text", ""),
            MAX_ARGUMENT_TEXT,
        )
        url = _safe_text(raw.get("url") or raw.get("final_url") or "", 2_048)
        normalized.append(
            {
                "id": source_id,
                "title": title,
                "excerpt": excerpt,
                "url": url,
                "status": "ok" if status in {"ok", "200", "success"} else "error",
            }
        )
    return normalized


def _extract_report_text(raw: Mapping[str, Any]) -> str:
    return _safe_text(
        raw.get("answer")
        or raw.get("raw_answer")
        or raw.get("report")
        or raw.get("text")
        or "",
        MAX_ARGUMENT_TEXT,
    )


def _normalize_reports(role_reports: Any) -> list[dict[str, Any]]:
    if isinstance(role_reports, (str, bytes, bytearray)) or not isinstance(role_reports, Sequence):
        return []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(list(role_reports)[:MAX_ARGUMENT_REPORTS]):
        if isinstance(raw, Mapping):
            answer = _extract_report_text(raw)
            question = _safe_text(raw.get("question", ""), MAX_ARGUMENT_TEXT)
            claim = _safe_text(raw.get("claim", ""), MAX_ARGUMENT_TEXT)
            counterclaim = _safe_text(raw.get("counterclaim", ""), MAX_ARGUMENT_TEXT)
            rebuttal = _safe_text(raw.get("rebuttal", ""), MAX_ARGUMENT_TEXT)
            cited = _bounded_strings(raw.get("cited_source_ids", raw.get("citedSourceIds", [])), maximum_items=MAX_ARGUMENT_EVIDENCE, maximum_text=MAX_ARGUMENT_SOURCE_ID)
            open_questions = _bounded_strings(raw.get("open_questions", raw.get("openQuestions", [])), maximum_items=8)
            agent_id = _safe_id(raw.get("agent_id", raw.get("id")), f"live-agent-{index + 1:02d}", 80)
            role = _safe_text(raw.get("role", "panel role"), 120) or "panel role"
            domain = _safe_text(raw.get("domain", "general"), 120) or "general"
            confidence = _bounded_float(raw.get("confidence"), 0.5)
            stance = _normalize_stance(raw.get("stance"), text=" ".join((answer, claim, counterclaim, rebuttal)))
        else:
            answer = _safe_text(raw, MAX_ARGUMENT_TEXT)
            question = ""
            claim = answer
            counterclaim = ""
            rebuttal = ""
            cited = []
            open_questions = []
            agent_id = f"live-agent-{index + 1:02d}"
            role = "panel role"
            domain = "general"
            confidence = 0.5
            stance = _normalize_stance(None, text=answer)

        if not answer and not claim and not counterclaim and not rebuttal:
            # Empty provider objects are not useful graph inputs.
            continue
        if not question:
            question = ""
        if not claim:
            claim = answer or counterclaim or "No claim supplied."
        if not counterclaim:
            counterclaim = "No counterclaim supplied."
        if not rebuttal:
            rebuttal = "No rebuttal supplied."
        normalized.append(
            {
                "agent_id": agent_id,
                "role": role,
                "domain": domain,
                "question": question,
                "answer": answer or claim,
                "claim": claim,
                "counterclaim": counterclaim,
                "rebuttal": rebuttal,
                "cited_source_ids": cited,
                "open_questions": open_questions,
                "stance": stance,
                "confidence": _rounded(confidence),
            }
        )
    return normalized


def _synthesis_outcome(synthesis: Any, question: str) -> str:
    if isinstance(synthesis, Mapping):
        outcome = _safe_text(synthesis.get("outcome", ""), MAX_ARGUMENT_TEXT)
        if outcome:
            return outcome
    if isinstance(synthesis, str):
        outcome = _safe_text(synthesis, MAX_ARGUMENT_TEXT)
        if outcome:
            return outcome
    return f"Bounded outcome assessment for: {question}"


def _node(node_id: str, node_type: str, label: str, **fields: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"id": node_id, "type": node_type, "label": _safe_text(label, MAX_ARGUMENT_LABEL)}
    result.update(fields)
    return result


def _edge(source: str, target: str, relationship: str, **fields: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": source,
        "target": target,
        "relationship": relationship,
    }
    result.update(fields)
    return result


def _add_edge(edges: list[dict[str, Any]], seen: set[tuple[str, str, str]], edge: dict[str, Any]) -> None:
    key = (str(edge["source"]), str(edge["target"]), str(edge["relationship"]))
    if key in seen or len(edges) >= MAX_ARGUMENT_EDGES:
        return
    seen.add(key)
    edges.append(edge)


def _build_scoring(
    evidence: Sequence[Mapping[str, Any]], reports: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid_source_ids = {
        str(item.get("id"))
        for item in evidence
        if item.get("status") == "ok" and isinstance(item.get("id"), str)
    }
    source_ids_casefold = {source_id.casefold(): source_id for source_id in valid_source_ids}
    total_weight = 0.0
    support_weight = 0.0
    opposition_weight = 0.0
    directions: list[float] = []
    agent_scores: list[dict[str, Any]] = []
    cited_union: set[str] = set()

    for report in reports:
        stance = str(report.get("stance", "mixed"))
        confidence = _bounded_float(report.get("confidence"), 0.5)
        cited_ids = [
            source_ids_casefold[item.casefold()]
            for item in report.get("cited_source_ids", [])
            if isinstance(item, str) and item.casefold() in source_ids_casefold
        ]
        cited_ids = list(dict.fromkeys(cited_ids))
        cited_union.update(cited_ids)
        evidence_count = len(cited_ids)
        evidence_factor = 1.0 + min(0.25, evidence_count * 0.05)
        weight = max(0.25, confidence) * evidence_factor
        total_weight += weight
        if stance == "supportive":
            support_weight += weight
            direction = 1.0
        elif stance == "opposed":
            opposition_weight += weight
            direction = 0.0
        else:
            support_weight += weight * 0.5
            opposition_weight += weight * 0.5
            direction = 0.5
        directions.append(direction)

        has_claim = bool(_safe_text(report.get("claim")))
        has_counterclaim = bool(_safe_text(report.get("counterclaim")))
        has_rebuttal = bool(_safe_text(report.get("rebuttal")))
        argument_quality = (
            0.30 * float(has_claim)
            + 0.20 * float(has_counterclaim)
            + 0.20 * float(has_rebuttal)
            + 0.15 * min(1.0, evidence_count / 2.0)
            + 0.15 * confidence
        )
        agent_scores.append(
            {
                "agent_id": _safe_id(report.get("agent_id"), f"live-agent-{len(agent_scores) + 1:02d}"),
                "role": _safe_text(report.get("role", "panel role"), 120) or "panel role",
                "domain": _safe_text(report.get("domain", "general"), 120) or "general",
                "evidence_count": evidence_count,
                "stance": stance,
                "confidence": _rounded(confidence),
                "support_score": _rounded(weight if stance == "supportive" else weight * 0.5 if stance == "mixed" else 0.0),
                "opposition_score": _rounded(weight if stance == "opposed" else weight * 0.5 if stance == "mixed" else 0.0),
                "argument_quality": _rounded(argument_quality),
                "cited_source_ids": cited_ids,
            }
        )

    if total_weight:
        support_score = support_weight / total_weight
        opposition_score = opposition_weight / total_weight
    else:
        support_score = opposition_score = 0.0

    if len(directions) < 2:
        disagreement = 0.0
    else:
        pair_count = 0
        difference = 0.0
        for index, left in enumerate(directions):
            for right in directions[index + 1 :]:
                difference += abs(left - right)
                pair_count += 1
        disagreement = difference / pair_count if pair_count else 0.0
    consensus = 1.0 - disagreement
    source_count = len(valid_source_ids)
    evidence_coverage = (len(cited_union) / source_count) if source_count else 0.0
    if not source_count and reports:
        evidence_coverage = 0.0

    scoring = {
        "evidence_coverage": _rounded(evidence_coverage),
        "support_score": _rounded(support_score),
        "opposition_score": _rounded(opposition_score),
        "consensus": _rounded(consensus),
        "disagreement": _rounded(disagreement),
        "method": {
            "name": "deterministic-bounded-argument-engine",
            "description": "Scores are derived from bounded source citations and structured role reports.",
            "provider_graph_fields_used": False,
        },
        "limits": [
            f"At most {MAX_ARGUMENT_EVIDENCE} evidence items and {MAX_ARGUMENT_REPORTS} role reports are scored.",
            "Only cited source IDs that match collected successful evidence count toward coverage.",
            "Stance and quality are bounded report signals, not verified truth or comprehensive internet data.",
        ],
    }
    return scoring, agent_scores


def build_argument_analysis(
    question: Any,
    evidence_items: Any = None,
    role_reports: Any = None,
    synthesis: Any = None,
    **aliases: Any,
) -> dict[str, Any]:
    """Build the deterministic argument map, scores, and per-agent scores."""

    if evidence_items is None:
        evidence_items = aliases.get("evidence", aliases.get("sources"))
    if role_reports is None:
        role_reports = aliases.get("reports", aliases.get("agent_reports"))
    if synthesis is None:
        synthesis = aliases.get("synthesis_report")
    normalized_question = _safe_text(question, MAX_ARGUMENT_TEXT) or "Unspecified question"
    evidence = _normalize_evidence(evidence_items)
    reports = _normalize_reports(role_reports)
    outcome = _synthesis_outcome(synthesis, normalized_question)

    nodes: list[dict[str, Any]] = [
        _node(
            "question",
            "central_question",
            normalized_question,
            question=normalized_question,
        )
    ]
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    source_node_ids: dict[str, str] = {}
    for source in evidence:
        source_id = str(source["id"])
        node_id = f"evidence-{source_id}"
        source_node_ids[source_id.casefold()] = node_id
        label = source.get("title") or source.get("excerpt") or source_id
        nodes.append(
            _node(
                node_id,
                "evidence",
                label,
                source_id=source_id,
                status=source.get("status", "error"),
                excerpt=_safe_text(source.get("excerpt", ""), MAX_ARGUMENT_TEXT),
                url=_safe_text(source.get("url", ""), 2_048),
            )
        )
        _add_edge(edges, seen_edges, _edge(node_id, "question", "informs"))

    claim_node_ids: list[tuple[str, str, str]] = []
    for index, report in enumerate(reports):
        agent_id = _safe_id(report.get("agent_id"), f"live-agent-{index + 1:02d}")
        claim_id = f"{agent_id}-claim"
        counterclaim_id = f"{agent_id}-counterclaim"
        claim_node_ids.append((agent_id, claim_id, counterclaim_id))
        claim_label = _safe_text(report.get("claim"), MAX_ARGUMENT_LABEL) or "No claim supplied."
        counter_label = _safe_text(report.get("counterclaim"), MAX_ARGUMENT_LABEL) or "No counterclaim supplied."
        nodes.append(
            _node(
                claim_id,
                "agent_claim",
                claim_label,
                agent_id=agent_id,
                role=_safe_text(report.get("role", "panel role"), 120),
                domain=_safe_text(report.get("domain", "general"), 120),
                stance=report.get("stance", "mixed"),
                confidence=_bounded_float(report.get("confidence"), 0.5),
                cited_source_ids=list(report.get("cited_source_ids", [])),
            )
        )
        nodes.append(
            _node(
                counterclaim_id,
                "agent_counterclaim",
                counter_label,
                agent_id=agent_id,
                role=_safe_text(report.get("role", "panel role"), 120),
                domain=_safe_text(report.get("domain", "general"), 120),
            )
        )
        _add_edge(edges, seen_edges, _edge("question", claim_id, "frames"))
        _add_edge(edges, seen_edges, _edge(claim_id, counterclaim_id, "answers_with_counterclaim"))
        stance = report.get("stance", "mixed")
        outcome_relationship = "supports" if stance == "supportive" else "opposes" if stance == "opposed" else "qualifies"
        _add_edge(edges, seen_edges, _edge(claim_id, "outcome", outcome_relationship))
        _add_edge(edges, seen_edges, _edge(counterclaim_id, "outcome", "challenges"))
        for cited_id in list(report.get("cited_source_ids", []))[:MAX_ARGUMENT_EVIDENCE]:
            source_node_id = source_node_ids.get(str(cited_id).casefold())
            if source_node_id:
                _add_edge(edges, seen_edges, _edge(source_node_id, claim_id, "supports"))

    outcome_id = "outcome"
    nodes.append(
        _node(
            outcome_id,
            "outcome",
            outcome,
            outcome=outcome,
        )
    )
    _add_edge(edges, seen_edges, _edge("question", outcome_id, "resolves_to"))

    scoring, agent_scores = _build_scoring(evidence, reports)
    argument_map = {
        "version": 1,
        "generated_locally": True,
        "nodes": nodes[:MAX_ARGUMENT_NODES],
        "edges": edges[:MAX_ARGUMENT_EDGES],
        "limits": {
            "max_nodes": MAX_ARGUMENT_NODES,
            "max_edges": MAX_ARGUMENT_EDGES,
            "max_evidence": MAX_ARGUMENT_EVIDENCE,
            "max_reports": MAX_ARGUMENT_REPORTS,
        },
    }
    return {
        "argument_map": argument_map,
        "scoring": scoring,
        "agent_scores": agent_scores,
    }


def build_argument_map(
    question: Any,
    evidence_items: Any = None,
    role_reports: Any = None,
    synthesis: Any = None,
    **aliases: Any,
) -> dict[str, Any]:
    """Return only the deterministic argument map for small integrations."""

    return build_argument_analysis(
        question, evidence_items, role_reports, synthesis, **aliases
    )["argument_map"]


def score_arguments(
    question: Any,
    evidence_items: Any = None,
    role_reports: Any = None,
    synthesis: Any = None,
    **aliases: Any,
) -> dict[str, Any]:
    """Return the deterministic scoring object for small integrations."""

    del question  # scoring currently depends on evidence and reports only.
    return build_argument_analysis(
        "", evidence_items, role_reports, synthesis, **aliases
    )["scoring"]


# Descriptive aliases for integrations that use either noun.
analyze_arguments = build_argument_analysis
construct_argument_map = build_argument_map


__all__ = [
    "MAX_ARGUMENT_EDGES",
    "MAX_ARGUMENT_EVIDENCE",
    "MAX_ARGUMENT_NODES",
    "MAX_ARGUMENT_REPORTS",
    "build_argument_analysis",
    "build_argument_map",
    "construct_argument_map",
    "analyze_arguments",
    "score_arguments",
]
