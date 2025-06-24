# Pydantic Reference

Quick reference for effective Pydantic usage, focused on practical patterns and common gotchas.

## Core Concepts

### BaseModel vs Dataclasses
```python
# Use BaseModel for data validation/serialization
class User(BaseModel):
    name: str
    age: int

# Use pydantic.dataclasses for stdlib compatibility  
from pydantic.dataclasses import dataclass

@dataclass
class Config:
    debug: bool = False
```

**Choose BaseModel when**: JSON serialization, API validation, schema generation  
**Choose dataclasses when**: Replacing stdlib dataclasses, lighter weight needs

## Feature Coverage

Quick assessment of Pydantic capabilities for "Does Pydantic support...?" questions.

### Validation & Parsing
- **Type validation**: All Python types + runtime enforcement
- **Type coercion**: Automatic conversion (lax mode) or strict type matching
- **Custom validators**: Field-level and model-level validation logic
- **Conditional validation**: Fields dependent on other field values
- **Multiple validation errors**: Collects all errors before raising
- **Validation modes**: Strict mode, assignment validation, partial validation

### Data Types
- **Standard types**: `int`, `float`, `str`, `bool`, `list`, `dict`, `set`, `tuple`
- **Date/time**: `datetime`, `date`, `time`, `timedelta`
- **Advanced types**: `UUID`, `Path`, `FilePath`, `DirectoryPath`, `Enum`
- **Typing support**: `Optional`, `Union`, `Literal`, `TypedDict`, `NamedTuple`
- **Pydantic types**: `EmailStr`, `HttpUrl`, `IPvAnyAddress`, `SecretStr`, `Json`
- **Constrained types**: `conint`, `constr`, `confloat` with bounds/patterns
- **Custom types**: Define your own types with validation logic

### Serialization & Deserialization
- **Dict conversion**: `model_dump()` with field inclusion/exclusion
- **JSON support**: `model_dump_json()` and `model_validate_json()`
- **Partial updates**: `exclude_unset=True` for sparse data
- **Field aliases**: Different names for serialization vs Python attributes
- **Custom serializers**: Override serialization for specific types
- **Computed fields**: Include derived values in serialized output

### Schema & Documentation
- **JSON Schema**: Generate OpenAPI-compatible schemas
- **Field documentation**: Descriptions, examples, constraints
- **Model documentation**: Schema titles, descriptions, examples
- **OpenAPI integration**: Automatic API documentation generation

### Advanced Features
- **Model inheritance**: Extend models with additional fields/validation
- **Generic models**: Type-parameterized models with `Generic[T]`
- **Forward references**: Handle circular dependencies with string annotations
- **Model composition**: Nested models and complex data structures
- **Root validators**: Validate entire model rather than individual fields
- **Pre/post processors**: Transform data before/after validation
- **Immutable models**: `frozen=True` for read-only instances
- **Model rebuilding**: Dynamic model updates and forward reference resolution

### Integration Capabilities
- **FastAPI**: Native integration for API validation
- **Pydantic Settings**: Environment variable and config file parsing
- **SQLAlchemy**: ORM integration via SQLModel
- **Dataclasses compatibility**: Drop-in replacement for stdlib dataclasses
- **Pytest fixtures**: Excellent testing integration
- **CLI libraries**: Automatic argument parsing from models

### Performance Features
- **Rust core**: Core validation written in Rust for speed
- **Lazy validation**: Validate only when needed
- **Model caching**: Reuse validation logic across instances
- **Efficient serialization**: Optimized JSON operations

## Key Features

### Automatic Validation
- Type annotations enforced at runtime
- Detailed error reporting with location info
- Type coercion in lax mode (default): `"123"` → `123`

### Built-in Serialization
```python
user = User(name="John", age=30)
user.model_dump()         # → dict
user.model_dump_json()    # → JSON string
User.model_validate(data) # dict → model
```

## Critical Patterns

### ⚠️ Computed Fields Must Use @computed_field
```python
from pydantic import BaseModel, computed_field

class Rectangle(BaseModel):
    width: int
    height: int
    
    # ❌ Won't appear in serialization
    @property
    def area(self) -> int:
        return self.width * self.height
    
    # ✅ Will appear in model_dump() and JSON
    @computed_field
    @property
    def area(self) -> int:
        return self.width * self.height
```

### Custom Validation
```python
from pydantic import field_validator

class User(BaseModel):
    name: str
    age: int
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if len(v) < 2:
            raise ValueError('Name too short')
        return v.strip()
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 0:
            raise ValueError('Age must be positive')
        return v
```

### Field Constraints
```python
from pydantic import Field

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, le=10000)
    tags: list[str] = Field(max_length=10)
```

## Error Handling

### Error Structure
```python
from pydantic import ValidationError

try:
    User(name="", age=-5)
except ValidationError as e:
    for error in e.errors():
        print(f"Field: {error['loc']}")
        print(f"Message: {error['msg']}")
        print(f"Type: {error['type']}")
        print(f"Input: {error['input']}")
```

### Multiple Errors
Pydantic collects ALL validation errors before raising, making debugging easier.

## Configuration

### Common Config Options
```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,    # Auto-strip strings
        validate_assignment=True,     # Validate on field assignment  
        extra='forbid',              # Reject extra fields
        frozen=True,                 # Make immutable
    )
```

## Testing Patterns

### What to Test
✅ **Test your code**:
- Custom validators (`@field_validator`)
- Computed fields (`@computed_field`)
- Business logic and edge cases
- Model interactions

❌ **Don't test the framework**:
- Basic type validation
- Built-in serialization
- Simple field assignments

### Validation Testing Pattern
```python
import pytest
from pydantic import ValidationError

def test_validator_accepts_valid_data():
    user = User(name="John", age=25)
    assert user.name == "John"

def test_validator_rejects_invalid_data():
    with pytest.raises(ValidationError) as exc_info:
        User(name="", age=25)
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]['loc'] == ('name',)
    assert 'too short' in errors[0]['msg']  # Use 'in' not '=='
```

### Computed Field Testing
```python
def test_computed_field():
    rect = Rectangle(width=3, height=4)
    
    # Test direct access
    assert rect.area == 12
    
    # Test serialization includes it
    data = rect.model_dump()
    assert data['area'] == 12
```

### Fixture Composition
```python
@pytest.fixture
def valid_user_data():
    return {"name": "John", "age": 25}

@pytest.fixture  
def user(valid_user_data):
    return User(**valid_user_data)

def test_user_behavior(user):
    assert user.name == "John"
```

## Common Use Cases

### API Models
```python
class CreateUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(ge=0, le=150)

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    
    @computed_field
    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.email})"
```

### Configuration
```python
class AppConfig(BaseModel):
    debug: bool = False
    database_url: str
    max_connections: int = Field(default=10, ge=1, le=100)
    
    @field_validator('database_url')
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v.startswith(('postgresql://', 'sqlite://')):
            raise ValueError('Invalid database URL')
        return v
```

### Data Processing
```python
class DataRecord(BaseModel):
    timestamp: datetime
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @computed_field
    @property
    def processed_value(self) -> float:
        return round(self.value * 1.1, 2)  # Apply 10% markup
```

## Type Support

### Standard Types
- Basic: `int`, `float`, `str`, `bool`
- Collections: `list`, `dict`, `set`, `tuple`
- Time: `datetime`, `date`, `time`
- Other: `UUID`, `Path`, `Enum`, `Optional`, `Union`

### Special Pydantic Types
- `EmailStr`, `HttpUrl`, `IPvAnyAddress`
- `SecretStr`, `SecretBytes` (for sensitive data)
- `Json` (for JSON strings)
- `Field()` with constraints

## Performance Tips

- Use `model_validate()` for dict→model conversion (faster than constructor)
- Consider `exclude_unset=True` for partial updates
- Use `frozen=True` for immutable models
- Profile with real data before optimizing

## Project Organization

### Recommended Structure
```
src/
├── models/
│   ├── __init__.py      # Export public models
│   ├── base.py          # Shared base classes
│   ├── user.py          # Domain models
│   └── api.py           # Request/response models
└── services/
    └── user_service.py  # Business logic
```

### Circular Import Handling
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_model import OtherModel

class MyModel(BaseModel):
    related: "OtherModel"  # String annotation

# After all models defined:
MyModel.model_rebuild()
```

## Key Gotchas

1. **Computed fields need `@computed_field`** - Regular `@property` won't serialize
2. **Error messages have prefixes** - Use `'expected' in error['msg']` not equality
3. **Validation runs on assignment** - Only with `validate_assignment=True`
4. **Mocking requires valid data** - Pydantic validates even mock objects
5. **Forward references need rebuilding** - Call `model_rebuild()` after definition

## Quick Reference

```python
from pydantic import BaseModel, Field, field_validator, computed_field

class ExampleModel(BaseModel):
    # Basic field
    name: str
    
    # Field with constraints
    age: int = Field(ge=0, le=150)
    
    # Optional with default
    active: bool = True
    
    # Custom validator
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip().title()
    
    # Computed field (will serialize)
    @computed_field
    @property
    def display_name(self) -> str:
        return f"{self.name} ({'active' if self.active else 'inactive'})"

# Usage
model = ExampleModel(name="john", age=25)
data = model.model_dump()  # Includes computed fields
json_str = model.model_dump_json()
```