# RFC 0011: Agent Improvement

| start_date   | 2026-07-28 |
| :----------- | :--------- |
| mlflow_issue | |
| rfc_pr       | |

| Author(s)              | [Nehanth Narendrula](https://github.com/Nehanth) (Red Hat) |
| :--------------------- | :-- |
| **Date Last Modified** | 2026-08-24 |
| **AI Assistant(s)**    | Claude Code |

**Table of contents**

- [Overview](#overview)
- [Part 1: Trace-Aware Webhook Events](#part-1-trace-aware-webhook-events)
- [Part 2: Improvement Workflow](#part-2-improvement-workflow)
- [User journeys](#user-journeys)
- [Why now](#why-now)

## Overview

MLflow can already observe agents in production. Tracing captures every step of execution, [automatic evaluations](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/) run LLM judges on live traces, and [issue detection](https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues/) clusters failures into named problems with severity and root-cause summaries. What is missing is what happens next. When quality drops, nothing notifies anyone, and the path from a detected issue to a fixed agent is entirely manual.

This RFC adds that missing half in two parts, each detailed in its own document in this directory: [Part 1](part-1-trace-webhooks.md) covers trace-aware webhook events, and [Part 2](part-2-improvement-workflow.md) covers the improvement workflow that builds on them. Following the current submission model, both keep implementation detail to a minimum; the design is presented through the problem, the user journeys, and the pieces of MLflow each part builds on.

## Part 1: Trace-Aware Webhook Events

[part-1-trace-webhooks.md](part-1-trace-webhooks.md)

MLflow's webhook system currently supports [15 event types](https://mlflow.org/docs/latest/self-hosting/webhooks/), and every one of them belongs to the model registry, prompt registry, or gateway budgets. The observability side of MLflow cannot notify anyone of anything: an evaluation score can collapse, issue detection can find a critical problem, traces can start erroring, and the only way to find out is to open the UI.

Part 1 extends the existing webhook system with events for exactly those situations: an evaluation score crossing a threshold, issue detection finding or updating an issue, and traces completing with errors. It reuses the webhook delivery infrastructure MLflow already ships and adds nothing else. It is useful on its own even if nothing from Part 2 is ever built.

## Part 2: Improvement Workflow

[part-2-improvement-workflow.md](part-2-improvement-workflow.md)

Part 2 is the full loop: a new **Improve** tab in the experiment UI. The user tells MLflow where the agent's code lives. When issue detection or an evaluation threshold flags a problem, the workflow analyzes the traces together with the source code, and a coding agent applies the fix and opens a pull request. A deprecated dependency that starts failing every trace becomes a PR the engineer reviews minutes later, with links back to the traces that triggered it. The fixed failure then becomes a regression eval so it cannot silently come back.

The fix itself is written by a coding agent, not by MLflow. [OpenCode](https://github.com/sst/opencode), an open-source coding agent, ships as the bundled default so the loop works out of the box — and the fix pipeline is three pluggable interfaces (acquire source, apply a fix, submit for review), so teams pick their own source platform, coding agent, and review flow. Inference always runs on the user's own model access — MLflow never runs managed inference for fixes, and nothing merges without a human review. The workflow rolls out in stages, starting read-only with diagnoses, and enables server-side fix execution once RFC-0002's remote executors provide isolation. Part 1 does not wait for any of it.

This gets more capable as MLflow's governance work lands. With the [MCP Registry (RFC-0004)](https://github.com/mlflow/rfcs/blob/main/rfcs/0004-mcp-registry/0004-mcp-registry.md), the [Skill Registry (RFC-0008)](https://github.com/mlflow/rfcs/blob/main/rfcs/0008-mvp-skill-registry/0008-mvp-skill-registry.md), [Extended Agent Plugins (PR #27)](https://github.com/mlflow/rfcs/pull/27), and an eventual agent registry, MLflow will know which MCP servers, skills, and repos make up an agent. The improvement workflow can then diagnose which component an issue actually lives in and open the fix against the right one — an agent's skill, its MCP server, or its own code — instead of requiring the user to wire that mapping up by hand.

## User journeys

**Journey 1 — alerting, bring your own pipeline.** A team runs a support agent in production with automatic evaluations scoring correctness on sampled traces. They register a webhook: if the rolling correctness average drops below 0.7, notify their endpoint. One Tuesday a model deprecation breaks a tool call and scores collapse. The webhook fires within minutes, lands in PagerDuty, and the on-call engineer investigates with a direct link to the failing traces. What they do next is up to them — MLflow's job was to make the signal available.

**Journey 2 — full self-healing with a coding agent.** The same team connects their GitHub repo in the Improve tab. The same failure occurs, but this time the error events trigger the improvement workflow: it pulls the failing traces, correlates the error against the source, and identifies the deprecated dependency. A coding agent gets the diagnosis and the trace context, updates the dependency, and opens a pull request whose description links every affected trace. The engineer reviews and merges. The failure is added to an evaluation dataset, so the next run of the eval suite would catch any regression. Total human effort: one code review.

Each part's document walks these journeys in detail, including the payloads, UI paths, and the prompt-drift and scheduled-detection variants.

## Why now

Alerting on production eval scores is standard across the ecosystem. [LangSmith](https://docs.langchain.com/langsmith/alerts), [Langfuse](https://langfuse.com/docs/metrics/features/monitors), [Arize](https://arize.com/docs/ax/machine-learning/machine-learning/how-to-ml/monitors/configure-monitors/notifications-and-integrations), [Braintrust](https://www.braintrust.dev/docs/guides/automations), [Datadog LLM Observability](https://docs.datadoghq.com/llm_observability/monitoring/), and [Galileo](https://docs.galileo.ai/how-to-guides/basics/set-up-alerts-on-logs) all ship it. MLflow has none of it, and neither does [Databricks' managed MLflow monitoring](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring).

The fix side is where the industry is converging right now. [LangSmith Engine](https://docs.langchain.com/langsmith/engine) detects recurring issues in production traces, diagnoses root cause against the connected source code, and proposes fixes as GitHub pull requests — but it is closed source, runs only on LangChain-managed inference, and is metered per scan. [Raindrop 2.0](https://www.raindrop.ai/blog/introducing-raindrop-2) launched under the banner "Self-Healing Agents": Raindrop detects and root-causes the failure, the customer's own coding agent pulls the context over MCP, fixes it, and opens the PR — and the company [raised $15M from Lightspeed](https://www.prnewswire.com/news-releases/raindrop-raises-15-million-to-detect-critical-ai-agent-failures-302628853.html) with Vercel, Replit, and Speak among its customers.

No open-source platform ships this loop. MLflow already has the tracing, the online judges, the issue detection, the prompt registry, and an ecosystem of coding agents to plug in. It is the best-positioned open-source project to be first, and most of the work is connecting pieces that already exist.
