---
theme: apple-basic
highlighter: shiki
class: text-left
paginate: true
background: black
color: white
layout: intro
---

# Chandy-Lamport 
## Snapshot Algorithm

<div class="absolute bottom-10">
  <span class="font-700">
    Hanani Bathina
  </span>
</div>

---
layout: statement
---

# Goal

## To compute **Snapshot** of an execution of a distributed algorithm.

---
layout: statement
---

# Constraints

## Develop a snapshot algorithm that works without freezing the execution of the basic algorithm of which the snapshot is taken. 

---
layout: statement
---

# Why is it useful?

## They are useful to try to determine offline properties that will remain true as soon as they have become true, such as deadlock, termination or garbage. 


_________________________________


## Moreover, snapshots can be used for checkpointing to restart after a failure, or for debugging.


---
layout: section
---

# Fokkin's POV

---
layout: default
---

# Configuration

The global state of a distributed algorithm, called a _configuration_, evolves by means of _transitions_. The overall behavior of a distributed system is captured by a _transition system_, which consists of
1. a set $\mathcal{C}$ of configurations
2. a binary transition relation $\rightarrow$ on $\mathcal{C}$, and
3. a set $\mathcal{I} \subseteq \mathcal{C}$ of initial configurations

A configuration $\gamma$ is _terminal_ if it has no outgoing transition $\gamma \rightarrow \delta$ for no $\delta \in \mathcal{C}$

A configuration $\delta$ is _reachable_ if there is $\gamma_0 \in \mathcal{I}$ and a sequence of $\gamma_0, \gamma_1, \gamma_2, ..., \gamma_k$ with $\gamma_i \rightarrow \gamma_{i+1}$ for all $0 \le i \lt k$ and $y_k = \delta$

---
layout: default
---

# Key Concepts

1. **Messages**: Messages of the basic algorithm are called _basic messages_ and messages of the snapshot algorithm are called _control messages_.
2. **Computation**: All permutations of concurrent events in an execution does not affect the result of the execution. These permutations together form a _computation_. All computations start in the same configuration, and if they are finite, they all end in the same terminal configuration.
3. **Complication**: Processes take local snapshots and compute channel states at different moments in time. Therefore, a snapshot may not actually represent a configuration of an ongoing execution.
4. **Solution**: But, A configuration of an execution in the same _computation_ is good enough. Such a snapshot is _consistent_. 


---
layout: default
---

# Examples of Bad/Inconsistent Snapshots

## The Ghost Message (Effect without a Cause)
Process $p$ could take a local snapshot and then send a basic message $m$ to $q$, where $q$ could either take a local snapshot after receipt of $m$ or include $m$ in the state of the channel $pq$. Here, $m$ is a ghost message.
### Why it's "Bad"
$m$ has become a "Ghost." It exists in $q$'s hand, but it was never born in $p$'s hand. This violates causality.

## The Missing Message (Lost in the Void)
A process $p$ could take a local snapshot after sending a basic message $m$. While $q$ could receive $m$ after taking its local snapshot and exclude $m$ from the channel state of $pq$. 

### Why it's "Bad"
Information is lost.

---
layout: center 
---
# Consistent Snapshots

A snapshot is **consistent** if:
* For each presnapshot even $a$, all the events that are causally before $a$ are also presnapshot.

* A basic message is included in a channel state $iff$ the corresponding send event is presnapshot and corresponding receive event is postsnapshot.

> The first property implies that the snapshot is a configuration of an execution that is in the same computation as the actual execution.

---
layout: default
---

# The Chandy-Lamport Algorithm

## Steps

* Any initiator can decide to take local snapshot of its state. It then sends a control message $<marker>$ through all its outgoing channels to let its neighbours take a snapshot too.

* When a process that has not yet taken a snapshot receives a $<marker>$ message, it takes a local snapshot of its state, and sends a $<marker>$ message through all its out-going channels. 

* A process _q_ computes as channel state for an incoming channel $pq$ the messages that it receives via $pq$ after taking its local snapshot and before receiving a $<marker>$ message from $p$. 

* The **Chandy-Lamport** algorithm terminates at a process when it has received a $<marker>$ message through all its incoming channels.

> The Computed Snapshot may not be a configuration of the actual execution. But it should be in the same computation as the actual execution.

---
layout: default
---

# Why does Chandy-Lamport work?
* The Chandy-Lamport algorithm guarantees that the computed snapshot is consistent.

## Condition 1: For each presnapshot even $a$, all the events that are causally before $a$ are also presnapshot.

* If $a$ and $b$ occur at the same process, then this is trivially the case.

* The interesting case is where $a$ is a send event and $b$ the corresponding receive event. 

> Suppose that $a$ occurs at process $p$ and $b$ at process $q$. Since $b$ is presnapshot, q has not yet received a $<marker>$ message at the time it performs $b$. Since channels are **FIFO**,
this implies $p$ has not yet sent a $<marker>$ message to $q$ at the time it performs $a$.
Hence, $a$ is presnapshot.

---
layout: default
---

# Why does Chandy-Lamport work?
* The Chandy-Lamport algorithm guarantees that the computed snapshot is consistent.

## Condition 2: A basic message is included in a channel state $iff$ the corresponding send event is presnapshot and corresponding receive event is postsnapshot.

* $q$ (has already taken snapshot) must receive $m$ before $<marker>$ through $pq$; 
* so since channels are FIFO, $p$ must send $m$ before $<marker>$ into $pq$.

---
layout: section
---

# Djikstra's POV

---
layout: section
---

# The Abstract Model
## "Atomic Actions"

---
layout: default
---

# Atomic Actions
A distributed computation is a succession of atomic actions:

1. **Change State:** The machine updates its internal memory.
2. **Accept:** It takes at most **one** message from an input buffer.
3. **Send:** It sends at most **one** message to an output buffer.

> **Crucial Rule:** Messages in a buffer *enable* an action but do not *force* it. However, every message will eventually be accepted.


---
layout: default
---

# Overarching Goal: Detecting Stability

The algorithm aims to collect state information to detect a **Stable Predicate**.

* **Definition:** A predicate is "stable" if, once it holds in some state, it holds in all possible later states.
* **Examples:** Termination, Deadlock, or Garbage Collection.
* **The Challenge:** In a distributed system, how do we "stop the clock" to check this without actually stopping the machines?

> The purpose of distributed snapshot algorithm of Chandy-Lamport is to collect such state information that, on account of it, stability can be detected.

---
layout: default
---

# States

The distributed snapshot algorithm is superimposed on the distributed algorithm such that, while the distrbuted algorithm evolves from state $s_0$ to state $s_1$, it collects the description of a so-called "snapshot state" $SSS$.

## Properties of $SSS$

There need not have been a single moment at which it occured - $SSS$ is a state that is possible after $S_0$ and $S_1$ is a state that is possible after $SSS$. 

Hence if stability has been reached in $S_0$, $SSS$ satisfied the stable predicate, and consequently $S_1$ satisifies it as well.

---
layout: section
---

# The Coloring Metaphor
## White vs. Red

---
layout: default
---

# The Snapshot Transition
Imagine every component in the system has a color.

* **Initial State ($S_0$):** All machines and all messages are **White**.
* **The Transition:** During the snapshot, each machine turns from **White** to **Red** exactly once.
* **Final State ($S_1$):** Eventually, all machines and all messages are **Red**.

### The Coloring Rules:
* An atomic action gets the color of the machine performing it.
* A message gets the color of the action that sends it.

---
layout: default
---

# The "Magic" Assumption
To ensure a consistent snapshot, we temporarily assume a bit of **magic**:

> **"No Red message is ever accepted in a White action."**

If we maintain this, the **Snapshot State ($SSS$)** consists of:
1. For each machine: Its state at the moment it turned from White to Red.
2. For each buffer: The string of White messages accepted by a Red machine.

---
layout: section
---

# The Logic of Consistency
## Why is a "Fake" state valid?

---
layout: default
---

# Commutativity & Equivalence
Dijkstra proves consistency by showing that we can reorder actions:

* **Observation:** Two actions from different machines commute *unless* one sends a message the other accepts.
* **The Logic:** Since no Red message is accepted by a White action, a **Red action** and a **subsequent White action** *always* commute.
* **The Result:** We can "bubble sort" the history. All White actions can be moved to the front, followed by all Red actions.

> And here all machines can turn red between the last white and the first red action; the system state at that moment is evidently the state yielded by the snapshot algorithm.  


---
layout: center
class: text-center
---

# The Result
$S_0 \rightarrow SSS \rightarrow S_1$

$SSS$ is a state that is possible after $S_0$.  
$S_1$ is a state that is possible after $SSS$.  

**If stability holds in $S_0$, it holds in $SSS$, and thus holds in $S_1$.**

---
layout: section
---

# Implementation
## Replacing "Magic" with Markers

---
layout: default
---

# Markers: The Physical Manifestation
How do we enforce the "No Red $\rightarrow$ White" rule without magic?

1. **Turning Red:** A machine turns Red the moment it accepts its first **Marker** (if it hasn't turned Red already).
2. **Sending Markers:** Upon turning Red, a machine sends a Marker over every output buffer *before* sending any further messages.
3. **FIFO Rule:** Markers follow the FIFO nature of the buffers. This ensures all White messages arrive before the Marker, and all Red messages arrive after.

---
layout: default
---

# Termination of the Snapshot
The algorithm is decentralized but finite.

* **Initiation:** One or more machines turn Red spontaneously.
* **Propagation:** Because every machine is reachable and messages are eventually accepted, the "Redness" spreads.
* **Completion:** A machine knows its local snapshot is done when it has received a Marker on **every** input buffer.
* **Collection:** Local data is sent to a central point (initiator) to evaluate the Stable Predicate.

---
layout: statement
---

# Summary
## The Chandy-Lamport algorithm doesn't capture a single moment in time. 
## It captures a **logically equivalent** state that preserves the causal truth of the system.