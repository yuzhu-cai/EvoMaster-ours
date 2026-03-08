import logging
from pathlib import Path
from typing import Any, Tuple
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from ..utils.code import read_code,save_code_to_file
import uuid
import os
from evomaster.agent import BaseAgent

try:
    from ..utils.grading import validate_submission
    _HAS_GRADING = True
except ImportError as e:
    _HAS_GRADING = False
    _GRADING_IMPORT_ERROR = str(e)

class DraftExp(BaseExp):
    def __init__(self, draft_agent, debug_agent, metric_agent, config,exp_name):
        super().__init__(draft_agent, config)
        self.draft_agent = draft_agent
        self.debug_agent = debug_agent
        self.metric_agent = metric_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.terminal_output = ""
        self.code = ""
        self.debug_times = 0
        self._exp_name = exp_name
        self.workspace_path = os.path.join(self.draft_agent.session.config.workspace_path, exp_name)

    @property
    def exp_name(self) -> str:
        """返回实验阶段名称"""
        return self._exp_name

    def _check_grading_valid(self, submission_path: str) -> Tuple[bool, str]:
        """使用 grading_server 校验 submission 格式。
        返回 (是否通过, 不通过时的理由，通过时为空字符串)。
        """
        if not _HAS_GRADING:
            return True, ""
        servers = getattr(self.config, "grading_servers", []) or []
        exp_id = getattr(self.config, "competition_id", None) or getattr(self.config, "exp_id", None)
        data_root = os.path.join(os.getcwd(), getattr(self.config, "data_root", None))
        if not servers or not exp_id:
            return True, ""
        ok, res = validate_submission(
            exp_id,
            Path(submission_path),
            server_urls=servers,
            dataset_root=data_root,
        )
        if not ok:
            reason = str(res) if res else "grading_server 调用失败"
            self.logger.warning(
                "grading_server 调用失败，默认视为 submission 格式合法通过: %s", reason
            )
            return True, ""
        if isinstance(res, dict) and not res.get("is_valid", True):
            reason = res.get("result") or res.get("details") or str(res)
            self.logger.warning("grading_server 格式校验未通过: %s", reason)
            return False, reason
        self.logger.info(f"grading_server 格式校验通过")
        return True, ""

    def run(self, task_description: str, data_preview: str, data_knowledge: str, model_knowledge: str, task_id: str = "exp_001") -> dict:

        self.logger.info("Starting draft task execution")
        self.logger.info(f"Task: {task_description}")

        try:
            while True:
                if self.draft_agent:
                    self.logger.info("=" * 60)
                    self.logger.info("Step 1: Draft Agent analyzing task...")
                    self.logger.info("=" * 60)
                    BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)
                    
                    draft_original_format_kwargs = self.draft_agent._prompt_format_kwargs.copy()
                    self.draft_agent._prompt_format_kwargs.update({
                        'task_description': task_description,
                        'data_preview': data_preview,
                        'data_knowledge': data_knowledge,
                        'model_knowledge': model_knowledge,
                    })

                    draft_task = TaskInstance(
                        task_id=f"{task_id}_draft",
                        task_type="draft",
                        description=task_description,
                        input_data={},
                    )

                    draft_trajectory = self.draft_agent.run(draft_task)
                    draft_result = self._extract_agent_response(draft_trajectory)
#                     draft_result = f"""
# ```python
# import pandas as pd
# import numpy as np
# import warnings
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score

# warnings.filterwarnings('ignore')

# print("Starting fast execution script...")

# # 1. Load data
# # ---------------------------------------------------------
# print("Loading data...")
# train_df = pd.read_csv('./input/train.csv')
# test_df = pd.read_csv('./input/test.csv')

# # 2. Clean text
# # ---------------------------------------------------------
# def clean_text(text):
#     if isinstance(text, str):
#         # Remove surrounding quotes if present
#         text = text.strip('"')
#         # Handle escaped characters
#         try:
#             text = bytes(text, 'utf-8').decode('unicode_escape', errors='ignore')
#         except:
#             pass
#         # Basic cleaning
#         text = ' '.join(text.split())  # Remove extra whitespace
#     else:
#         text = "" # Handle NaN
#     return text

# print("Cleaning text...")
# train_df['clean_comment'] = train_df['Comment'].apply(clean_text)
# test_df['clean_comment'] = test_df['Comment'].apply(clean_text)

# # 3. Vectorization (TF-IDF) - This replaces the BERT Tokenizer
# # ---------------------------------------------------------
# print("Vectorizing text (TF-IDF)...")
# # max_features=10000 限制特征数量，保证速度极快
# vectorizer = TfidfVectorizer(
#     stop_words='english', 
#     max_features=10000, 
#     ngram_range=(1, 2)
# )

# # Fit on train, transform train and test
# X_train_all = vectorizer.fit_transform(train_df['clean_comment'])
# X_test = vectorizer.transform(test_df['clean_comment'])
# y_train_all = train_df['Insult'].values

# # 4. Split train data for validation
# # ---------------------------------------------------------
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_all,
#     y_train_all,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_train_all
# )

# # 5. Model Training (Logistic Regression) - Replaces BERT Model
# # ---------------------------------------------------------
# print("Training Logistic Regression model...")
# # n_jobs=-1 uses all CPU cores
# model = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
# model.fit(X_train, y_train)

# # 6. Validation
# # ---------------------------------------------------------
# print("Validating...")
# val_preds = model.predict_proba(X_val)[:, 1] # Get probability for class 1
# auc_score = roc_auc_score(y_val, val_preds)
# print(f'Validation AUC: {{auc_score:.4f}}')

# # 7. Prediction on Test Set
# # ---------------------------------------------------------
# print("Predicting on test set...")
# test_preds = model.predict_proba(X_test)[:, 1]

# # 8. Create submission file
# # ---------------------------------------------------------
# submission_df = pd.DataFrame({{
#     'Insult': test_preds
# }})

# # Ensure predictions are in [0, 1] range (Logic Regression implies this, but good practice)
# submission_df['Insult'] = submission_df['Insult'].clip(0, 1)

# # Save submission
# os.makedirs('./submission', exist_ok=True)
# submission_path = './submission/submission.csv'
# submission_df.to_csv(submission_path, index=False)
# print(f"Submission saved to {{submission_path}} with {{len(submission_df)}} predictions")

# # Also save to working directory for backup
# os.makedirs('./working', exist_ok=True)
# submission_df.to_csv('./working/submission.csv', index=False)

# print("Sample predictions (first 5):")
# print(submission_df.head())
# print("Done! Execution completed in seconds.")
# ```
# """
                    draft_code,self.code = read_code(draft_result, self.uid)
                    save_code_to_file(self.workspace_path, "run.py", draft_code)
                    tool_call_obj = ChatCompletionMessageToolCall(
                        id="call_123",
                        type="function",
                        function=Function(
                            name="execute_bash",
                            arguments='{"command": "python run.py","timeout": "86400"}'
                        )
                    )
                    observation, info =self.draft_agent._execute_tool(tool_call_obj)
                    self.terminal_output = observation
                    submission_path = os.path.join(self.workspace_path, "submission", f"submission_{self.uid}.csv")
                    if info.get("exit_code") == 0 and os.path.exists(submission_path):
                        grading_ok, grading_reason = self._check_grading_valid(submission_path)
                        is_success = grading_ok
                        if not grading_ok:
                            self.terminal_output = (
                                f"{self.terminal_output}\n\n"
                                "[grading] 代码成功运行，但提交格式不合法。grading_server 校验结果: "
                                f"{grading_reason}"
                            )
                    else:
                        is_success = False
                    self.logger.info(f"Draft Agent execute_bash result: {observation}")
                    self.logger.info(f"Draft Agent execute_bash info: {info}")

                    
                    self.logger.info("Draft completed")
                    self.logger.info(f"Draft result: {draft_result[:2000]}...")
                    self.draft_agent._prompt_format_kwargs = draft_original_format_kwargs


                if self.metric_agent and is_success:
                    self.logger.info("=" * 60)
                    self.logger.info("Step 2: Metric Agent executing task...")
                    self.logger.info("=" * 60)
                    metric_original_format_kwargs = self.metric_agent._prompt_format_kwargs.copy()
                    self.metric_agent._prompt_format_kwargs.update({
                        'terminal_output': observation
                    })
                    metric_task = TaskInstance(
                        task_id=f"{task_id}_metric",
                        task_type="metric",
                        input_data={},
                    )

                    metric_trajectory = self.metric_agent.run(metric_task)
                    metric_result = self._extract_agent_response(metric_trajectory)
                    try:
                        validation_score = float(metric_result.split("\\boxed{")[1].split("}")[0])
                    except:
                        is_success = False
                        validation_score = None
                    self.logger.info(f"validation score: {validation_score}")
                    self.logger.info("Metric completed")
                    self.logger.info(f"Metric result: {metric_result[:2000]}...")
                    self.metric_agent._prompt_format_kwargs = metric_original_format_kwargs
                
                debug_times = 0
                while is_success==False and debug_times < 3:
                    self.logger.info("=" * 60)
                    self.logger.info("Step 3: Debug Agent executing task...")
                    self.logger.info("=" * 60)
                    debug_original_format_kwargs = self.debug_agent._prompt_format_kwargs.copy()
                    self.debug_agent._prompt_format_kwargs.update({
                        'task_description': task_description,
                        'terminal_output': self.terminal_output,
                        'buggy_code': self.code,
                        'data_preview': data_preview,
                    })
                    debug_task = TaskInstance(
                        task_id=f"{task_id}_debug",
                        task_type="debug",
                        task_description=task_description,
                        input_data={},
                    )
                    debug_trajectory = self.debug_agent.run(debug_task)
                    debug_result = self._extract_agent_response(debug_trajectory)
                    debug_code,self.code = read_code(debug_result, self.uid)
                    save_code_to_file(self.workspace_path, "run.py", debug_code)
                    tool_call_obj = ChatCompletionMessageToolCall(
                        id="call_123",
                        type="function",
                        function=Function(
                            name="execute_bash",
                            arguments='{"command": "python run.py","timeout": "86400"}'
                        )
                    )
                    observation, info =self.debug_agent._execute_tool(tool_call_obj)
                    self.terminal_output = observation
                    submission_path = os.path.join(self.workspace_path, "submission", f"submission_{self.uid}.csv")
                    if info.get("exit_code") == 0 and os.path.exists(submission_path):
                        grading_ok, grading_reason = self._check_grading_valid(submission_path)
                        debug_success = grading_ok
                        if not grading_ok:
                            self.terminal_output = (
                                f"{self.terminal_output}\n\n"
                                "[grading] 代码成功运行，但提交格式不合法。grading_server 校验结果: "
                                f"{grading_reason}"
                            )
                    else:
                        debug_success = False
                    self.logger.info(f"Debug Agent execute_bash result: {observation}")
                    self.logger.info(f"Debug Agent execute_bash info: {info}")
                    self.logger.info("Debug completed")
                    self.logger.info(f"Debug result: {debug_result[:2000]}...")
                    self.debug_agent._prompt_format_kwargs = debug_original_format_kwargs

                    if self.metric_agent and debug_success:
                        self.logger.info("=" * 60)
                        self.logger.info("Step 4: Metric Agent executing task...")
                        self.logger.info("=" * 60)
                        metric_original_format_kwargs = self.metric_agent._prompt_format_kwargs.copy()
                        self.metric_agent._prompt_format_kwargs.update({
                            'terminal_output': observation
                        })
                        metric_task = TaskInstance(
                            task_id=f"{task_id}_metric",
                            task_type="metric",
                            input_data={},
                        )

                        metric_trajectory = self.metric_agent.run(metric_task)
                        metric_result = self._extract_agent_response(metric_trajectory)
                        try:
                            validation_score = float(metric_result.split("\\boxed{")[1].split("}")[0])
                        except:
                            debug_success = False
                            validation_score = None
                        self.logger.info(f"validation score: {validation_score}")
                        self.logger.info("Metric completed")
                        self.logger.info(f"Metric result: {metric_result[:2000]}...")
                        self.metric_agent._prompt_format_kwargs = metric_original_format_kwargs

                    if debug_success:
                        is_success = True
                        validation_score = validation_score
                        return is_success, validation_score, self.uid,self.code
                    else:
                        is_success = False
                        validation_score = None
                        debug_times += 1

                return is_success, validation_score, self.uid, self.code

        except Exception as e:
            self.logger.error(f"Draft task execution failed: {e}", exc_info=True)
            raise ValueError(f"Draft task execution failed: {e}")




