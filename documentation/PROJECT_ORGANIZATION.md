# Project Organization and AI Assistant Guidelines

**Created:** 2025-11-26 16:22:00  
**Last Updated:** 2025-11-26 16:22:00

Guide for maintaining consistent project organization and enabling AI assistants to follow project conventions.

## Overview

This project uses several mechanisms to ensure AI assistants follow consistent conventions:

1. **`.cursorrules`** - Cursor IDE-specific AI rules (most important)
2. **`.ai-conventions`** - Detailed conventions reference
3. **`AI_README.md`** - Quick reference for AI assistants

## Directory Structure

```
subcorticalSTRF/
├── .cursorrules                    # Cursor AI rules (READ THIS FIRST)
├── .ai-conventions                 # Detailed project conventions
├── AI_README.md                    # Quick AI assistant guide
├── tests/                          # ALL test files go here
│   ├── test_loggers_refactored.py
│   ├── test_wsr_refactored.py
│   └── test_*.py
├── documentation/                  # ALL documentation goes here
│   ├── QUICK_REFERENCE_LOGGERS.md
│   ├── WSR_MODEL_REFACTORING.md
│   └── *.md
├── loggers/                        # Logging and folder management
├── models/                         # Model implementations
│   ├── WSRmodel/
│   └── BEZ2018/
├── analyses/                       # Analysis results
└── model_comparisons/              # Model comparison outputs
```

## Critical Rules for AI Assistants

### 1. File Placement (ALWAYS ENFORCE)

❌ **NEVER DO THIS:**
```python
# Creating test in root
create_file("/path/to/project/test_something.py", ...)

# Creating docs in root  
create_file("/path/to/project/SOME_GUIDE.md", ...)
```

✅ **ALWAYS DO THIS:**
```python
# Tests go in tests/
create_file("/path/to/project/tests/test_something.py", ...)

# Docs go in documentation/
create_file("/path/to/project/documentation/SOME_GUIDE.md", ...)
```

### 2. File Headers (ALWAYS INCLUDE)

Every new file must have a header with timestamp:

**Python:**
```python
"""
Module description.

Created: 2025-11-26 16:20:00
Author: Ekim Celikay (with AI assistance)
"""
```

**Markdown:**
```markdown
# Document Title

**Created:** 2025-11-26 16:20:00  
**Last Updated:** 2025-11-26 16:20:00

Description here.
```

### 3. Use Loggers Module (ALWAYS)

Never write manual logging setup. Use:
```python
from loggers import FolderManager, LoggingConfigurator, MetadataSaver
```

See `QUICK_REFERENCE_LOGGERS.md` for examples.

## How to Make AI Assistants Follow These Rules

### For Cursor IDE Users

The `.cursorrules` file is automatically read by Cursor AI. Just ensure it exists in the project root.

### For Other AI Assistants

Include this in your system prompt or project context:
```
Before creating any files, read .cursorrules and .ai-conventions in the project root.
Always place test files in tests/ and documentation in documentation/.
Always add timestamp headers to new files.
```

### For ChatGPT / Claude / Other Chat AIs

At the start of each session, provide:
```
I'm working on the subcorticalSTRF project. Please follow these rules:
1. Test files → tests/ directory
2. Documentation → documentation/ directory  
3. Add timestamp headers to all new files
4. Use the loggers module for output management
5. Read .cursorrules for full conventions
```

## Verification Checklist

When an AI assistant creates files, verify:

- [ ] Test files are in `tests/` directory
- [ ] Documentation files are in `documentation/` directory
- [ ] File has proper header with timestamp
- [ ] Python files use `loggers` module (not manual setup)
- [ ] Imports are in correct order (stdlib, third-party, local)
- [ ] Logger used instead of print() statements

## Benefits of This System

1. **Consistency** - All AI assistants follow same conventions
2. **Traceability** - Timestamps show when files were created
3. **Organization** - Tests and docs in correct locations
4. **Maintainability** - Easier to find and update files
5. **Reusability** - loggers module reduces code duplication

## Updating Conventions

When updating conventions:

1. Update `.cursorrules` (primary source)
2. Update `.ai-conventions` (detailed reference)
3. Update this document
4. Update `AI_README.md` if needed
5. Commit all changes together

## Examples

### Creating a New Test

```python
"""
Test suite for new feature.

Created: 2025-11-26 16:30:00
Purpose: Verify new feature works correctly
"""
import pytest
from your_module import your_function

def test_your_function():
    \"\"\"Test that your_function works.\"\"\"
    result = your_function()
    assert result == expected
```

File path: `tests/test_new_feature.py`

### Creating Documentation

```markdown
# New Feature Guide

**Created:** 2025-11-26 16:30:00  
**Last Updated:** 2025-11-26 16:30:00

How to use the new feature.

## Usage
...
```

File path: `documentation/NEW_FEATURE_GUIDE.md`

## Troubleshooting

**Problem:** AI created file in wrong location

**Solution:** 
1. Move file to correct location
2. Remind AI to read `.cursorrules`
3. Consider adding reminder to system prompt

**Problem:** AI didn't add timestamp header

**Solution:**
1. Manually add header
2. Remind AI about header requirement
3. Update `.cursorrules` to make it more explicit

**Problem:** AI wrote manual logging setup

**Solution:**
1. Refactor to use loggers module
2. Point AI to `QUICK_REFERENCE_LOGGERS.md`
3. Emphasize "ALWAYS use loggers" in `.cursorrules`

## See Also

- `.cursorrules` - Primary AI rules file
- `.ai-conventions` - Detailed conventions
- `AI_README.md` - Quick start for AI
- `documentation/QUICK_REFERENCE_LOGGERS.md` - Loggers usage guide
