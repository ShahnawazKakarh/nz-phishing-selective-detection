"""NZ-localized feature extractors.

Detects NZ-specific phishing cues:
- Impersonated NZ entities (IRD, NZ Post, Waka Kotahi, NZ banks)
- .govt.nz / .co.nz domain spoofing and homoglyphs
- NZ phone-number formats (+64, 02x mobile prefixes)
- Te reo Māori salutations and codeswitching
- NZ currency, postcodes, and address patterns
"""
