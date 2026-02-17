# **PRD: Improving Metaworld hand insert PPO Training Performance**

## **Objective**

Improve the PPO policy performance on the **hand insert** task.
The goal is to increase success rates **without modifying the environment evaluation procedure**, while achieving:

* More stable training behavior
* Stronger convergence
* Reproducible performance gains

---

## **Ground Rules (Agent Contract)**

* **Environment must remain unchanged:**
  No modification to any Metaworld environment code or success criteria.

* **Training budget fixed:**
  Total training steps cannot be increased (must keep the original `total_steps` scale).

* **Modification scope restricted:**
  Only approved source files may be edited.

---

## **Codebase Path**

The entire codebase has already been copied into your workspace.
Your current working directory is the root of this codebase.

You may inspect files using `file_editor` or `terminal`.


### Related Code

You may modify **only** the following files:
1. **Network code**
    `./metaworld_algorithms/rl/networks.py`
    
2. **Network config**
    `./metaworld_algorithms/config/networks.py`

3. **Reinforcement Learning algorithm**
   `./metaworld_algorithms/rl/algorithms/ppo.py`

4. **Training script**
   `./examples/single_task/ppo_hand_insert.py`

### ⚠️ CRITICAL: How to Make Changes

**USE YOUR `file_editor` TOOL TO DIRECTLY MODIFY FILES!**

You have access to the following tools:
- `file_editor` - Use this to view and edit files directly, `old_str` is required for command: str_replace.
- `terminal` - Use this to run commands and inspect the codebase
- `task_tracker` - Use this to track your progress
## Tool Call Format
**You MUST use the standard structured tool calling interface to invoke tools.**
- Use formal tool calls provided by the API.
- DO NOT embed tool calls as plain text in your response unless you are unable to use the structured interface.
"""

### Example Tool Calls (Native Format):
Your tool calls should be handled by the model's native function calling mechanism.

**1. To view the file:**
- tool: `file_editor`
- arguments: {{ "command": "view", "path": "{codebase_dir}/policy/DP/diffusion_policy/config/robot_dp_14.yaml", "security_risk": "LOW" }}

**2. To modify a file:**
IMPORTANT: Please strictly follow the calling format to modify a file, or it will cause fatal erros! Espeacially, the old_str parameter is needed. You can view the original file to get old_str.
- tool: `file_editor`
- arguments: {{ 
    "command": "str_replace", 
    "path": "{codebase_dir}/policy/DP/diffusion_policy/config/robot_dp_14.yaml",
    "old_str": "lr: 1.0e-4",
    "new_str": "lr: 5.0e-5",
    "security_risk": "MEDIUM" 
  }}

--- 
**DO NOT** write Python scripts to modify files. Instead:

1. **Use `file_editor` to view the current code**
2. **Use `file_editor` to make your modifications directly**
3. **The system will automatically run training after you're done**

---

## **Example Workflow**

### **Step 1: Inspect current code**

Open:

```bash
./metaworld_algorithms/rl/networks.py
```

### **Step 2: Apply modifications**

For example:

* Increase network size

### **Step 3: Verify changes**

Re-open modified files via `file_editor`.

### **Step 4: Finish**

The system will automatically execute:

```bash
cd /workspace/codebase
bash train.sh
```

---

## **What You Should Do**

### **DO**

- Use `file_editor` to read files in `./codebase/`
- Use `file_editor` to modify the allowed files directly
- Use `terminal` to explore the codebase structure
- Explain your changes and rationale

### **DO NOT**

- Write Python scripts that modify files programmatically
- Import torch, tensorflow, or create training loops
- Call training scripts manually (the system does this automatically)
- Create new files outside the allowed list
- Run any python files, you can only edit the files.
- Do not make overly large changes to the hyperparameters in the PPO algorithm. The current configuration has already been carefully tuned by humans, and excessively large modifications may instead lead to a significant degradation in performance.
---

## **Possible Optimization Directions**
Mainly focus on these directions. You should first consider them insted of other improvements
1.  **Select Appropriate Network Depth and Width:** Generally, the network should not be too deep; you can, however, appropriately increase the network width. 
2. Adjust learning rate accordingly.
3. Do not make overly large changes to the hyperparameters in the PPO algorithm. The current configuration has already been carefully tuned by humans, and excessively large modifications may instead lead to a significant degradation in performance.
---


All modifications must preserve PPO’s external interface and expected behavior for other components depending on this module.


## **Deliverables**

After completing modifications, you must provide:

### **1. Change Summary**

* List modified files
* Specify exact locations + content of changes

### **2. Rationale**

* Explain why each change was made

### **3. Expected Impact**

* Explain the predicted improvements in performance or stability
