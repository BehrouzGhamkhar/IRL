Perfect — that actually puts you in a **great position**.
ML-Agents is *much easier* when you already know **Unity** *and* **RL in Python**.

I’ll teach you **ML-Agents from scratch**, but **compact, practical, and no fluff**.
Think of this as a **mental model + checklist** you can reuse forever.

---

# 🧠 What ML-Agents REALLY is (1-minute mental model)

**ML-Agents = RPC bridge between Unity and Python**

* Unity = **Environment + Physics + Observations + Rewards**
* Python = **Policy (PPO / SAC / etc.)**
* ML-Agents just serializes:

  * observations
  * rewards
  * done flags
  * actions

Nothing magical.

```
Unity (C# Agent)
    ↓ obs, reward, done
Python (Trainer)
    ↓ action
Unity
```

If either side sends garbage → **freeze / NaNs / timeout**.

---

# 🧩 Core ML-Agents components (memorize this)

### Unity side (YOU write this)

1. `Agent` (C# script)
2. `CollectObservations()`
3. `OnActionReceived()`
4. `AddReward()` / `EndEpisode()`
5. `Behavior Parameters` (Inspector)

### Python side

* Either:

  * **mlagents-learn** (built-in PPO)
  * OR **your own PyTorch code** (what you’re doing)

---

# 🧪 Minimal ML-Agents Agent (C#)

This is the **canonical skeleton**.
Everything else is variations.

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class SimpleAgent : Agent
{
    public Rigidbody rb;

    public override void OnEpisodeBegin()
    {
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.localPosition = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(rb.velocity);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float x = actions.ContinuousActions[0];
        float z = actions.ContinuousActions[1];

        Vector3 force = new Vector3(x, 0, z);
        rb.AddForce(force * 10f);

        AddReward(-0.001f); // step penalty
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        a[0] = Input.GetAxis("Horizontal");
        a[1] = Input.GetAxis("Vertical");
    }
}
```

---

# ⚙️ Behavior Parameters (CRITICAL)

This is where 90% of beginners break things.

In **Inspector → Behavior Parameters**:

| Field              | Value            |
| ------------------ | ---------------- |
| Behavior Name      | `SimpleBehavior` |
| Behavior Type      | **Default**      |
| Actions            | **Continuous**   |
| Continuous Actions | **2**            |
| Vector Observation | **6**            |
| Stacked Vectors    | **1**            |

⚠️ If **any of these mismatch Python**, Unity will hang.

---

# 🔍 Observations rule (VERY important)

### Rule of thumb

> **Observations must ALWAYS be finite and bounded**

### ❌ BAD

```csharp
sensor.AddObservation(transform.position); // world coords explode
```

### ✅ GOOD

```csharp
sensor.AddObservation(transform.localPosition / 5f);
sensor.AddObservation(rb.velocity / 10f);
```

### 🔥 Golden rule

If you see:

* NaN
* Infinity
* Very large numbers

👉 PPO will die silently.

---

# 🎯 Rewards (simple and safe)

Start with **tiny, dense rewards**.

### Example

```csharp
float distance = Vector3.Distance(transform.localPosition, target.localPosition);
AddReward(-distance * 0.001f);
```

### Episode termination

```csharp
if (distance < 0.5f)
{
    AddReward(1.0f);
    EndEpisode();
}
```

---

# 🔁 Step loop (what actually happens)

Per Unity physics step:

```
CollectObservations()
↓
Python receives obs
↓
Python sends action
↓
OnActionReceived()
↓
AddReward()
↓
done? EndEpisode()
```

If **OnActionReceived throws an exception** → Unity freezes.

---

# 🐍 Python side (two options)

## Option A — Built-in trainer (recommended for learning)

```bash
mlagents-learn config/ppo.yaml --run-id=test1
```

Press ▶ in Unity.

You get:

* Stable PPO
* TensorBoard
* No math bugs

👉 **Use this first** to validate your Unity environment.

---

## Option B — Custom Python (what you’re doing)

Your code is fine *conceptually*, but ML-Agents adds constraints:

### MUST DO

* Clip actions to `[-1, 1]`
* Validate no NaNs before sending
* Match behavior name EXACTLY
* Handle terminal vs decision steps correctly

---

# 🛡️ Unity-side NaN protection (DO THIS)

Add this inside `OnActionReceived`:

```csharp
if (float.IsNaN(x) || float.IsInfinity(x))
{
    Debug.LogError("Invalid action received!");
    EndEpisode();
    return;
}
```

This alone prevents freezes.

---

# 🧯 Why Unity freezes instead of errors

* Unity ML-Agents runs in **FixedUpdate**
* Physics + NaN = internal loop stall
* Python waits → timeout → kill

So:

> **Freeze = Unity exception or NaN**

Always check Unity console first.

---

# 🧠 Debugging checklist (save this)

### If Unity freezes:

1. Check Unity Console
2. Check action values
3. Check observation magnitudes
4. Set `time_scale = 1`
5. Use Heuristic mode

---

# 🚀 Learning path (FAST)

### Day 1

* One cube
* Move to target
* Built-in PPO

### Day 2

* Add obstacles
* Add sparse reward
* Curriculum

### Day 3

* Swap trainer for custom PyTorch
* Multi-agent
* Self-play

---

# 🎓 Key takeaway

ML-Agents is:

* **Unity first**
* **RL second**
* **Numerical stability always**

You already know the hard parts — now you just need to **respect ML-Agents’ constraints**.

---

If you want next:

* 🔥 A *known-good* Unity scene template
* 🔥 A minimal PPO Python script that NEVER freezes Unity
* 🔥 Reward shaping patterns that actually work

Tell me what you want next and we’ll build it step-by-step 👌
