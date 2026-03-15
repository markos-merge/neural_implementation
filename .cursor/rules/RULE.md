# Project rules

Add project-wide rules and conventions here. The agent will consider these when working in this repo.

## Implementation responsibility

- **The user must implement the functionality.** The AI assistant will **guide**—explaining concepts, suggesting next steps, answering questions, and reviewing code—but will **not** write the implementation.
- Provide explanations, formulas, hints, and pointers to theory or resources.
- Ask the user to try implementing before giving more detail.
- Code reviews focus on understanding and correctness, not on providing ready-made solutions.

## Code formatting and autocomplete

- **Autocomplete and generated code must adhere to `.clang-format`** in the project root.
- All suggestions, completions, and edits must follow the clang-format rules: tabs for indentation, spaces for alignment, east const, spaces inside parentheses, brace style, etc.
- When generating or editing code, format output to match `.clang-format` without requiring a separate format pass.

## C++ style

- **East const**: write `const` to the right of what it qualifies (e.g. `int const*`, `char const*`, `Foo const&`), not west const (`const int*`).
- **Member variables** must use the `m_` prefix (e.g. `m_size`, `m_data`).
- **Spaces in parentheses**: always one space inside `( )` (e.g. `( x )`, `if ( cond )`).
- **Access specifiers** (`public`, `private`, `protected`) must be indented one tab inside the class body.
- **Class members** must be indented one tab in from their access specifier (i.e. two tabs from the class body).
- **Template declarations** must always be on their own line above the class/struct/function (e.g. `template <typename T>\nclass Foo`, not `template <typename T> class Foo`).
- **Namespace braces** must be on the same line as the namespace (e.g. `namespace neural {`, not `namespace neural\n{`).

## Markdown and documentation

- **Math notation** in markdown must use the widely supported `$...$` (inline) and `$$...$$` (block) syntax, not `\(...\)` or `\[...\]`.
