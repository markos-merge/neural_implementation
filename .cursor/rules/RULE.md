# Project rules

Add project-wide rules and conventions here. The agent will consider these when working in this repo.

## C++ style

- **East const**: write `const` to the right of what it qualifies (e.g. `int const*`, `char const*`, `Foo const&`), not west const (`const int*`).
- **Member variables** must use the `m_` prefix (e.g. `m_size`, `m_data`).
- **Spaces in parentheses**: always one space inside `( )` (e.g. `( x )`, `if ( cond )`).
