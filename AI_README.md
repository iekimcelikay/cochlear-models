# AI Assistant Instructions

**Created:** 2025-11-26 16:22:00

⚠️ **IMPORTANT**: Before creating any files, please read `.ai-conventions` in this directory.

## Quick Rules

### File Placement
- ✅ Test files → `tests/`
- ✅ Documentation → `documentation/`
- ✅ Never create test/doc files in root

### File Headers
All new files must include:
- **Python files**: Docstring with creation date/time
- **Markdown files**: Header with created/updated timestamps
- **Test files**: Purpose statement

### Example Headers

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
# Title

**Created:** 2025-11-26 16:20:00  
**Last Updated:** 2025-11-26 16:20:00

Description here.
```

## Using the Loggers Module

Always use instead of manual setup:
```python
from loggers import FolderManager, LoggingConfigurator, MetadataSaver
```

See `documentation/QUICK_REFERENCE_LOGGERS.md` for examples.

## Full Conventions

Read `.ai-conventions` for complete guidelines.
