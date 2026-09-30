---
id: DOC-AGENT-001
type: policy
status: active
title: Agent Use Protocol
created: 2026-09-30
updated: 2026-09-30
---
# Agent Use Protocol

This document defines how the root agent should use subagents in this project.

The root agent owns the task end-to-end. Subagents are temporary workers used only when delegation materially improves execution.

Persistent project state belongs in the existing `progress/` system, not in agent conversational memory.

---

# 1. Core Rules

The root agent is responsible for:

- understanding the user's goal;
- reading relevant project instructions and state;
- deciding whether delegation is useful;
- defining subagent scope;
- integrating results;
- resolving conflicts;
- verifying the final outcome;
- updating relevant `progress/` state.

Delegating work does not delegate responsibility.

Subagent output is advisory until validated and integrated by the root agent.

Agents are temporary. Repository state is persistent.

---

# 2. Task Initialization

Before substantial work, inspect only the context necessary to understand the task.

Relevant context may include:

- repository instructions;
- current working tree;
- relevant source files and tests;
- configuration and documentation;
- relevant files under `progress/`;
- existing implementation patterns.

Do not read the entire repository or all progress history by default.

Understand the task before attempting to divide it.

---

# 3. Delegation Gate

For every task, the correct number of subagents may be:

- zero;
- one;
- multiple.

Do not create subagents merely because they are available.

Delegate only when doing so materially improves one or more of:

- parallelism;
- context isolation;
- specialized investigation;
- independent verification;
- ownership of a clearly separable workstream.

Prefer root-agent execution when:

- the task is small or already clear;
- work is tightly sequential;
- delegation overhead is comparable to doing the work directly;
- agents would repeatedly modify the same state;
- the root agent already has sufficient context;
- the remaining work is primarily integration.

Use the minimum useful number of agents.

---

# 4. Subagent Scope

Every subagent must receive a bounded mission.

Specify when relevant:

- objective;
- relevant context or paths;
- constraints;
- whether edits are allowed;
- expected output;
- completion criteria.

Prefer narrowly scoped tasks such as:

> Investigate the authentication refresh failure. Focus on `auth/` and related tests. Do not modify files. Return the likely root cause, evidence, affected files, and recommended fix.

Avoid open-ended assignments such as:

> Explore the project and fix whatever seems wrong.

Subagents should not silently expand their scope when unexpected issues appear. They should report the discovery to the root agent.

---

# 5. Common Agent Roles

Use roles as temporary execution modes, not permanent identities.

**Explorer**

Inspect code, trace behavior, locate relevant files, and identify constraints.

**Investigator**

Reproduce bugs, test hypotheses, isolate causes, and collect evidence.

**Implementer**

Perform a clearly bounded code change with explicit ownership.

**Reviewer**

Independently inspect a change for defects, regressions, incorrect assumptions, and missing tests.

**Verifier**

Run tests or other checks to determine whether the intended behavior is actually satisfied.

Create only the roles that materially help the task.

---

# 6. Parallelism and File Ownership

Parallelize only workstreams that are sufficiently independent.

Good candidates include:

- separate subsystems;
- independent investigations;
- implementation areas with clear file ownership;
- review or verification that does not interfere with active edits.

Avoid concurrent modification of the same files.

When several agents may need overlapping code, prefer:

1. parallel investigation;
2. root-agent integration and implementation.

If implementation is delegated, establish ownership boundaries whenever practical.

Strongly dependent work should run sequentially.

---

# 7. Project State

The existing `progress/` directory is the canonical persistent task-management state.

Agents may read relevant progress files when needed.

The root agent should normally own updates to shared progress state.

Subagents should not independently rewrite global project status unless explicitly assigned to do so.

Update `progress/` when work materially changes project state, such as when:

- a task or milestone completes;
- a blocker is discovered;
- an important assumption changes;
- implementation strategy changes;
- a durable decision is made;
- unfinished work must be resumed later.

Do not store temporary reasoning, routine investigation details, or conversational transcripts.

Persistent state should help a future agent determine:

- current goals;
- completed work;
- remaining work;
- important decisions;
- blockers or risks;
- next steps.

---

# 8. Result Contract

Subagents should return concise, decision-useful results.

When applicable, include:

**Outcome**  
What was concluded or completed.

**Evidence**  
Relevant files, commands, tests, observations, or other support.

**Changes**  
Files or behavior modified.

**Risks / Uncertainty**  
Anything not fully verified.

**Recommended Next Step**  
What the root agent should do with the result.

Prefer compact findings over long narratives.

---

# 9. Integration and Disagreement

The root agent must integrate results rather than concatenate them.

Integration includes:

- comparing findings;
- detecting contradictions;
- checking project constraints;
- validating important claims;
- reconciling overlapping recommendations;
- completing missing work.

When agents disagree:

1. isolate the disputed claim;
2. compare evidence;
3. inspect primary project evidence;
4. perform a targeted check if necessary;
5. decide at the root level.

Do not resolve disagreement by majority vote.

Evidence outranks agent count.

---

# 10. Verification

Do not treat an implementation as complete because a subagent reports success.

Perform verification proportional to risk.

Possible checks include:

- reading the diff;
- targeted tests;
- broader tests when justified;
- type or static checks;
- runtime validation;
- interface inspection;
- edge-case review;
- comparison against the original request.

Use an independent reviewer or verifier when the change is large, risky, subtle, or crosses important subsystem boundaries.

Do not create review agents automatically for trivial work.

---

# 11. Agent Hierarchy

Subagents should not normally spawn their own subagents.

Keep orchestration shallow:

Root Agent  
→ bounded subagents

Avoid unnecessary manager layers or deep agent trees.

Nested delegation is acceptable only when the delegated work is itself large enough to justify further decomposition and ownership remains clear.

Complex orchestration is a cost, not a goal.

---

# 12. Context Efficiency

Use subagents partly to protect root-agent context.

Do not duplicate large files, repository history, or unnecessary context into every agent prompt.

Provide:

- the objective;
- important constraints;
- known relevant paths;
- enough context to begin.

Let the subagent inspect details within its assigned scope.

The root agent should retain strategic context while subagents handle local investigative context.

---

# 13. Failure Handling

If a subagent fails, produces weak evidence, becomes blocked, or exceeds its scope, the root agent should decide whether to:

- finish the work directly;
- narrow and retry;
- delegate to another agent;
- use another investigative approach;
- abandon that workstream.

Do not repeatedly spawn agents without incorporating what was learned from earlier attempts.

---

# 14. Default Workflow

For each task:

1. Understand the request.
2. Read relevant project instructions and state.
3. Identify unknowns and separable workstreams.
4. Decide whether delegation materially helps.
5. If not, work directly.
6. If yes, create the minimum useful number of bounded subagents.
7. Run independent work in parallel when safe.
8. Collect and validate results.
9. Integrate findings and complete the task.
10. Verify the final outcome.
11. Update relevant `progress/` state.
12. Report the result.

---

# 15. Priority Order

When tradeoffs exist, prioritize:

1. correctness;
2. user intent;
3. project constraints;
4. verification;
5. simplicity;
6. context and agent efficiency;
7. parallelism.

Parallelism is useful only when it preserves the priorities above.

---

# Final Principle

Use agents to reduce complexity, not create it.

The root agent should always remain able to explain:

- why each subagent was created;
- what it was responsible for;
- what evidence it produced;
- which conclusions were accepted or rejected;
- how the final result was verified.

If the orchestration becomes harder to reason about than the task itself, simplify it.