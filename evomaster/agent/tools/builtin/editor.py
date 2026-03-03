"""EvoMaster Editor 工具

提供文件查看、创建、编辑的能力。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from ..base import BaseTool, BaseToolParams, ToolError, ToolParameterError

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


# 截断提示
TEXT_FILE_TRUNCATED_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. '
    'You should retry this tool after you have searched inside the file with `grep -n` in order to find '
    'the line numbers of what you are looking for.</NOTE>'
)
DIRECTORY_TRUNCATED_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of this directory has been shown to you. '
    'You should use `ls -la` instead to view large directories incrementally.</NOTE>'
)

# 每次编辑显示的上下文行数
SNIPPET_LINES = 4
MAX_OUTPUT_SIZE = 16000


def maybe_truncate(content: str, max_size: int = MAX_OUTPUT_SIZE, notice: str = TEXT_FILE_TRUNCATED_NOTICE) -> str:
    """中间截断内容"""
    if len(content) <= max_size:
        return content
    half = max_size // 2
    return content[:half] + "\n" + notice + "\n" + content[-half:]


class EditorToolParams(BaseToolParams):
    """Custom editing tool for viewing, creating and editing files in plain-text format.
    
    * State is persistent across command calls and discussions with the user
    * If `path` is a text file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
    * The `create` command cannot be used if the specified `path` already exists as a file
    * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
    * The `undo_edit` command will revert the last edit made to the file at `path`

    Before using this tool:
    1. Use the view tool to understand the file's contents and context
    2. Verify the directory path is correct (only applicable when creating new files)

    When making edits:
       - Ensure the edit results in idiomatic, correct code
       - Do not leave the code in a broken state
       - Always use absolute file paths (starting with /)

    CRITICAL REQUIREMENTS FOR USING THIS TOOL:

    1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file. The tool will fail if `old_str` doesn't match exactly.

    2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
       - Include sufficient context before and after the change point (3-5 lines recommended)
       - If not unique, the replacement will not be performed

    3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.
    """
    
    name: ClassVar[str] = "str_replace_editor"

    command: Literal["view", "create", "str_replace", "insert", "undo_edit"] = Field(
        description="The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`."
    )
    path: str = Field(
        description="Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.",
    )
    file_text: str = Field(
        default="",
        description="Required parameter of `create` command, with the content of the file to be created.",
    )
    old_str: str = Field(
        default="",
        description="Required parameter of `str_replace` command containing the string in `path` to replace.",
    )
    new_str: str = Field(
        default="",
        description="Optional parameter of `str_replace` command containing the new string. Required parameter of `insert` command containing the string to insert.",
    )
    insert_line: int = Field(
        default=-1,
        description="Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
    )
    view_range: list[int] = Field(
        default_factory=list,
        description="Optional parameter of `view` command when `path` points to a file. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start.",
    )


class EditorTool(BaseTool):
    """文件编辑工具"""
    
    name: ClassVar[str] = "str_replace_editor"
    params_class: ClassVar[type[BaseToolParams]] = EditorToolParams

    def __init__(self):
        super().__init__()
        # 文件编辑历史 {path: [(content, encoding), ...]}
        self._file_history: dict[str, list[tuple[str, str]]] = {}

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """执行编辑操作"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}
        
        assert isinstance(params, EditorToolParams)
        
        try:
            # 验证路径
            path_type = self._validate_path(session, params.command, params.path)
            
            if params.command == "view":
                return self._view(session, params.path, params.view_range, path_type)
            elif params.command == "create":
                return self._create(session, params.path, params.file_text)
            elif params.command == "str_replace":
                return self._str_replace(session, params.path, params.old_str, params.new_str)
            elif params.command == "insert":
                return self._insert(session, params.path, params.insert_line, params.new_str)
            elif params.command == "undo_edit":
                return self._undo_edit(session, params.path)
            else:
                return f"Unknown command: {params.command}", {}
        except ToolError as e:
            return f"ERROR:\n{str(e)}", {"error": str(e)}

    def _validate_path(
        self,
        session: BaseSession,
        command: str,
        path: str,
    ) -> Literal["file", "dir", "not_exist"]:
        """验证路径"""
        # 检查是否是绝对路径
        if not Path(path).is_absolute():
            raise ToolParameterError("path", path, "The path should be an absolute path, starting with `/`.")
        
        # 检查路径类型（优先检查目录，因为目录检查更可靠）
        if session.is_directory(path):
            path_type = "dir"
        elif session.is_file(path):
            path_type = "file"
        elif session.path_exists(path):
            # 如果路径存在但既不是文件也不是目录，再次检查
            # 可能是符号链接或其他特殊类型，尝试判断实际类型
            if session.is_directory(path):
                path_type = "dir"
            elif session.is_file(path):
                path_type = "file"
            else:
                # 未知类型，默认当作文件处理（但会在使用时再次检查）
                path_type = "file"
        else:
            path_type = "not_exist"
        
        # 验证命令与路径类型的兼容性
        if path_type == "not_exist" and command != "create":
            raise ToolParameterError("path", path, f"The path {path} does not exist.")
        
        # 对于 create 命令，需要更严格的检查
        if command == "create":
            # 再次确认路径不存在（防止误判）
            if session.is_file(path):
                raise ToolParameterError("path", path, f"File already exists at: {path}. Cannot overwrite files using command `create`.")
            if session.is_directory(path):
                raise ToolParameterError("path", path, f"The path {path} is a directory. Cannot create a file with the same name as a directory.")
            if session.path_exists(path):
                # 路径存在但不是文件也不是目录，可能是其他类型（如符号链接）
                raise ToolParameterError("path", path, f"Path already exists at: {path}. Cannot overwrite using command `create`.")
        
        if path_type == "dir" and command != "view":
            raise ToolParameterError("path", path, f"The path {path} is a directory and only the `view` command can be used on directories.")
        
        return path_type

    def _view(
        self,
        session: BaseSession,
        path: str,
        view_range: list[int],
        path_type: Literal["file", "dir", "not_exist"],
    ) -> tuple[str, dict[str, Any]]:
        """查看文件或目录"""
        # 再次检查路径类型，确保判断正确（防止 path_type 误判）
        if path_type == "dir" or session.is_directory(path):
            if view_range:
                raise ToolParameterError("view_range", view_range, "The `view_range` parameter is not allowed for directories.")
            
            # 列出目录内容（最多 2 层）
            result = session.exec_bash(f"find -L {path} -maxdepth 2 -not -path '*/\\.*' | head -500 | sort")
            output = result.get("stdout", "")
            output = maybe_truncate(output, max_size=MAX_OUTPUT_SIZE, notice=DIRECTORY_TRUNCATED_NOTICE)
            
            return f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{output}", {}
        
        # 读取文件
        content = session.read_file(path)
        init_line = 1
        
        # 处理 view_range
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolParameterError("view_range", view_range, "It should be a list of two integers.")
            
            lines = content.rstrip("\n").split("\n")
            n_lines = len(lines)
            start, end = view_range
            
            if start < 1 or start > n_lines:
                raise ToolParameterError("view_range", view_range, f"Start line {start} is out of range [1, {n_lines}].")
            if end != -1:
                if end < start:
                    raise ToolParameterError("view_range", view_range, f"End line {end} should be >= start line {start}.")
                if end > n_lines:
                    raise ToolParameterError("view_range", view_range, f"End line {end} exceeds file length {n_lines}.")
            
            if end == -1:
                content = "\n".join(lines[start - 1:])
            else:
                content = "\n".join(lines[start - 1:end])
            init_line = start
        
        return self._format_output(content, path, init_line), {}

    def _create(self, session: BaseSession, path: str, file_text: str) -> tuple[str, dict[str, Any]]:
        """创建文件"""
        session.write_file(path, file_text)
        self._file_history[path] = [(file_text, "utf-8")]
        return f"File created successfully at: {path}", {}

    def _str_replace(
        self,
        session: BaseSession,
        path: str,
        old_str: str,
        new_str: str,
    ) -> tuple[str, dict[str, Any]]:
        """替换字符串"""
        if new_str == old_str:
            raise ToolParameterError("new_str", new_str, "No replacement was performed. `new_str` and `old_str` must be different.")
        
        content = session.read_file(path)
        
        # 查找所有匹配
        pattern = re.escape(old_str)
        matches = list(re.finditer(pattern, content))
        
        if not matches:
            # 尝试 strip 后重试
            old_str_stripped = old_str.strip()
            new_str_stripped = (new_str or "").strip()
            pattern = re.escape(old_str_stripped)
            matches = list(re.finditer(pattern, content))
            
            if matches:
                # 检查 strip 后是否相同
                if old_str_stripped == new_str_stripped:
                    raise ToolParameterError("new_str", new_str, "No replacement was performed. `new_str` and `old_str` must be different (after stripping whitespace).")
                old_str = old_str_stripped
                new_str = new_str_stripped
            else:
                raise ToolError(f"No replacement was performed, old_str did not appear verbatim in {path}.")
        
        if len(matches) > 1:
            # 计算行号
            line_numbers = sorted(set(content.count("\n", 0, m.start()) + 1 for m in matches))
            raise ToolError(f"No replacement was performed. Multiple occurrences of old_str in lines {line_numbers}. Please ensure it is unique.")
        
        # 执行替换
        match = matches[0]
        replacement_line = content.count("\n", 0, match.start()) + 1
        new_content = content[:match.start()] + new_str + content[match.end():]
        
        # 保存历史并写入
        if path not in self._file_history:
            self._file_history[path] = []
        self._file_history[path].append((content, "utf-8"))
        session.write_file(path, new_content)
        
        # 创建代码片段
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n") + 1
        snippet = "\n".join(new_content.split("\n")[start_line:end_line + 1])
        
        msg = f"The file {path} has been edited. "
        msg += self._format_output(snippet, f"a snippet of {path}", start_line + 1)
        msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
        
        return msg, {}

    def _insert(
        self,
        session: BaseSession,
        path: str,
        insert_line: int,
        new_str: str,
    ) -> tuple[str, dict[str, Any]]:
        """插入内容"""
        content = session.read_file(path)
        lines = content.rstrip("\n").split("\n")
        n_lines = len(lines)
        
        if insert_line < 0 or insert_line > n_lines:
            raise ToolParameterError("insert_line", insert_line, f"It should be within the range [0, {n_lines}]")
        
        # 插入新行
        new_lines = new_str.split("\n")
        result_lines = lines[:insert_line] + new_lines + lines[insert_line:]
        new_content = "\n".join(result_lines)
        
        # 保存历史并写入
        if path not in self._file_history:
            self._file_history[path] = []
        self._file_history[path].append((content, "utf-8"))
        session.write_file(path, new_content)
        
        # 创建代码片段
        start_line = max(0, insert_line - SNIPPET_LINES + 1)
        end_line = insert_line + SNIPPET_LINES + 1
        snippet_lines = lines[start_line:insert_line] + new_lines + lines[insert_line:end_line]
        snippet = "\n".join(snippet_lines)
        
        msg = f"The file {path} has been edited. "
        msg += self._format_output(snippet, "a snippet of the edited file", start_line + 1)
        msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
        
        return msg, {}

    def _undo_edit(self, session: BaseSession, path: str) -> tuple[str, dict[str, Any]]:
        """撤销编辑"""
        if path not in self._file_history or not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")
        
        old_content, old_encoding = self._file_history[path].pop()
        session.write_file(path, old_content, old_encoding)
        
        return f"Last edit to {path} undone successfully. {self._format_output(old_content, path)}", {}

    def _format_output(self, content: str, descriptor: str, init_line: int = 1) -> str:
        """格式化输出（添加行号）"""
        content = maybe_truncate(content, max_size=MAX_OUTPUT_SIZE)
        numbered_lines = [
            f"{i + init_line:6}\t{line}"
            for i, line in enumerate(content.split("\n"))
        ]
        return f"Here's the result of running `cat -n` on {descriptor}:\n" + "\n".join(numbered_lines) + "\n"

