"""Code indexer for extracting functions, classes, and documentation from codebases.

This module provides functionality to scan source code files and extract
meaningful entities (functions, classes, methods) along with their metadata
for storage in the code memory system.
"""

import ast
import os
import re
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime
import hashlib


class CodeIndexer:
    """Indexer for extracting code entities from source files.

    This class supports parsing multiple programming languages and extracting:
    - Function and method definitions
    - Class definitions
    - Docstrings and comments
    - Import statements
    - Dependencies between entities

    Attributes:
        supported_languages: Set of file extensions that can be indexed.

    Example:
        >>> indexer = CodeIndexer()
        >>> entities = indexer.index_file("src/utils.py")
        >>> for entity in entities:
        ...     print(f"{entity['entity_name']}: {entity['signature']}")
    """

    def __init__(self):
        """Initialize the code indexer."""
        self.supported_languages = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp"
        }

    def index_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Index a single source file and extract all code entities.

        Args:
            file_path: Path to the source file to index.

        Returns:
            List of entity dictionaries, each containing:
            - file_path: Path to the source file
            - entity_name: Name of the function/class
            - code_snippet: The actual code
            - docstring: Documentation string (if available)
            - signature: Function/method signature
            - language: Programming language
            - dependencies: List of other entities referenced
            - imports: List of import statements
            - complexity: Estimated complexity
            - last_modified: File modification timestamp

        Example:
            >>> indexer = CodeIndexer()
            >>> entities = indexer.index_file("example.py")
            >>> len(entities) > 0
            True
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()
        language = self.supported_languages.get(extension)

        if language is None:
            return []  # Unsupported file type

        # Get file modification time
        last_modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            # Return empty list if file can't be read
            return []

        # Parse based on language
        if language == "python":
            return self._index_python(file_path, content, last_modified)
        elif language in ["javascript", "typescript"]:
            return self._index_javascript(file_path, content, last_modified, language)
        else:
            # For other languages, use basic regex extraction
            return self._index_generic(file_path, content, last_modified, language)

    def index_directory(
        self,
        directory: str,
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """Index all supported files in a directory.

        Args:
            directory: Path to the directory to index.
            exclude_patterns: List of glob patterns to exclude (e.g., ["*/tests/*", "*.test.py"]).
            recursive: Whether to recursively index subdirectories. Default: True

        Returns:
            List of all extracted entity dictionaries from all files.

        Example:
            >>> indexer = CodeIndexer()
            >>> entities = indexer.index_directory(
            ...     "src/",
            ...     exclude_patterns=["*/tests/*", "*/__pycache__/*"]
            ... )
            >>> len(entities)
            42
        """
        exclude_patterns = exclude_patterns or []
        all_entities = []

        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Get all files
        if recursive:
            files = path.rglob("*")
        else:
            files = path.glob("*")

        for file_path in files:
            if not file_path.is_file():
                continue

            # Check if file should be excluded
            if self._should_exclude(str(file_path), exclude_patterns):
                continue

            # Check if file type is supported
            if file_path.suffix.lower() not in self.supported_languages:
                continue

            # Index the file
            try:
                entities = self.index_file(str(file_path))
                all_entities.extend(entities)
            except Exception:
                # Skip files that can't be indexed
                continue

        return all_entities

    def _should_exclude(self, file_path: str, patterns: List[str]) -> bool:
        """Check if a file path matches any exclude pattern."""
        from fnmatch import fnmatch
        for pattern in patterns:
            if fnmatch(file_path, pattern):
                return True
        return False

    def _index_python(
        self,
        file_path: str,
        content: str,
        last_modified: str
    ) -> List[Dict[str, Any]]:
        """Index a Python file using AST parsing.

        Args:
            file_path: Path to the file.
            content: File content as string.
            last_modified: ISO timestamp of last modification.

        Returns:
            List of extracted entity dictionaries.
        """
        entities = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []  # Can't parse file with syntax errors

        # Extract imports
        imports = self._extract_python_imports(tree)

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                entity = self._extract_python_function(node, content, file_path, imports, last_modified)
                if entity:
                    entities.append(entity)

            elif isinstance(node, ast.ClassDef):
                entity = self._extract_python_class(node, content, file_path, imports, last_modified)
                if entity:
                    entities.append(entity)

        return entities

    def _extract_python_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from a Python AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        return imports

    def _extract_python_function(
        self,
        node: ast.FunctionDef,
        content: str,
        file_path: str,
        imports: List[str],
        last_modified: str
    ) -> Optional[Dict[str, Any]]:
        """Extract information about a Python function."""
        # Get function name
        name = node.name

        # Get docstring
        docstring = ast.get_docstring(node) or ""

        # Get code snippet
        code_lines = content.split("\n")
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        code_snippet = "\n".join(code_lines[start_line:end_line])

        # Build signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"

        signature = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {name}({', '.join(args)}){returns}"

        # Extract dependencies (function calls within this function)
        dependencies = self._extract_python_dependencies(node)

        # Estimate complexity
        complexity = self._estimate_complexity(code_snippet)

        return {
            "file_path": file_path,
            "entity_name": name,
            "code_snippet": code_snippet,
            "docstring": docstring,
            "signature": signature,
            "language": "python",
            "dependencies": dependencies,
            "imports": imports,
            "complexity": complexity,
            "last_modified": last_modified
        }

    def _extract_python_class(
        self,
        node: ast.ClassDef,
        content: str,
        file_path: str,
        imports: List[str],
        last_modified: str
    ) -> Optional[Dict[str, Any]]:
        """Extract information about a Python class."""
        name = node.name
        docstring = ast.get_docstring(node) or ""

        # Get code snippet
        code_lines = content.split("\n")
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        code_snippet = "\n".join(code_lines[start_line:end_line])

        # Build signature
        bases = [ast.unparse(base) for base in node.bases]
        signature = f"class {name}({', '.join(bases)})" if bases else f"class {name}"

        # Extract method names as dependencies
        dependencies = [
            method.name for method in node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        complexity = self._estimate_complexity(code_snippet)

        return {
            "file_path": file_path,
            "entity_name": name,
            "code_snippet": code_snippet,
            "docstring": docstring,
            "signature": signature,
            "language": "python",
            "dependencies": dependencies,
            "imports": imports,
            "complexity": complexity,
            "last_modified": last_modified
        }

    def _extract_python_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls from a Python function node."""
        dependencies = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(child.func.attr)

        return list(set(dependencies))  # Remove duplicates

    def _index_javascript(
        self,
        file_path: str,
        content: str,
        last_modified: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Index a JavaScript/TypeScript file using regex patterns.

        Note: This is a basic implementation. For production use, consider
        using a proper JavaScript parser like esprima or babel.
        """
        entities = []

        # Extract imports
        import_pattern = r"import\s+.*?from\s+['\"](.+?)['\"]"
        imports = re.findall(import_pattern, content)

        # Extract function declarations
        func_pattern = r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*{[\s\S]*?}"
        for match in re.finditer(func_pattern, content):
            name = match.group(1)
            code_snippet = match.group(0)

            entities.append({
                "file_path": file_path,
                "entity_name": name,
                "code_snippet": code_snippet,
                "docstring": "",
                "signature": f"function {name}(...)",
                "language": language,
                "dependencies": [],
                "imports": imports,
                "complexity": self._estimate_complexity(code_snippet),
                "last_modified": last_modified
            })

        # Extract arrow functions assigned to const/let/var
        arrow_pattern = r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"
        for match in re.finditer(arrow_pattern, content):
            name = match.group(1)

            entities.append({
                "file_path": file_path,
                "entity_name": name,
                "code_snippet": match.group(0) + " ...",
                "docstring": "",
                "signature": f"const {name} = (...) => ...",
                "language": language,
                "dependencies": [],
                "imports": imports,
                "complexity": "medium",
                "last_modified": last_modified
            })

        # Extract class definitions
        class_pattern = r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*{[\s\S]*?}"
        for match in re.finditer(class_pattern, content):
            name = match.group(1)
            code_snippet = match.group(0)

            entities.append({
                "file_path": file_path,
                "entity_name": name,
                "code_snippet": code_snippet,
                "docstring": "",
                "signature": f"class {name}",
                "language": language,
                "dependencies": [],
                "imports": imports,
                "complexity": self._estimate_complexity(code_snippet),
                "last_modified": last_modified
            })

        return entities

    def _index_generic(
        self,
        file_path: str,
        content: str,
        last_modified: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Generic indexer for unsupported languages using basic patterns."""
        # For now, return empty list
        # This could be extended with basic regex patterns for other languages
        return []

    def _estimate_complexity(self, code: str) -> str:
        """Estimate code complexity based on simple heuristics.

        Args:
            code: The code snippet to analyze.

        Returns:
            Complexity estimate: "low", "medium", or "high".
        """
        lines = len(code.split("\n"))
        control_flow = len(re.findall(r"\b(if|for|while|switch|case|catch)\b", code))
        nested_depth = max(
            (len(line) - len(line.lstrip())) // 4
            for line in code.split("\n")
        ) if code else 0

        score = lines / 10 + control_flow * 2 + nested_depth

        if score < 5:
            return "low"
        elif score < 15:
            return "medium"
        else:
            return "high"

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file for change detection.

        Args:
            file_path: Path to the file.

        Returns:
            Hexadecimal hash string.

        Example:
            >>> indexer = CodeIndexer()
            >>> hash1 = indexer.compute_file_hash("example.py")
            >>> # Modify file
            >>> hash2 = indexer.compute_file_hash("example.py")
            >>> hash1 != hash2
            True
        """
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        return sha256.hexdigest()
