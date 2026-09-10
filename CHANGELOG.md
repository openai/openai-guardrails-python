# Changelog

## [0.3.3](https://github.com/openai/openai-guardrails-python/compare/v0.3.2...v0.3.3) (2026-09-10)


### Bug Fixes

* bound secret-key extension matching work ([#111](https://github.com/openai/openai-guardrails-python/issues/111)) ([36e37b5](https://github.com/openai/openai-guardrails-python/commit/36e37b509087c68b12c5c449d7c582e32f57d6a2))
* enforce component restrictions for IP URL allow-list entries ([#116](https://github.com/openai/openai-guardrails-python/issues/116)) ([8bd0fab](https://github.com/openai/openai-guardrails-python/commit/8bd0fabb1aaac830ddaa7ccf4dbd09b7927f27b8))
* honor strict errors in prompt injection detection ([#114](https://github.com/openai/openai-guardrails-python/issues/114)) ([41f0195](https://github.com/openai/openai-guardrails-python/commit/41f0195fa7cf4d9cb18af64bf1ed36481da87566))
* mask repeated encoded PII in structured content ([#112](https://github.com/openai/openai-guardrails-python/issues/112)) ([accdd48](https://github.com/openai/openai-guardrails-python/commit/accdd48596dbc41faa850bab6fe7c4005abeb17a))
* Match safety identifier endpoints by hostname ([#100](https://github.com/openai/openai-guardrails-python/issues/100)) ([6d837ec](https://github.com/openai/openai-guardrails-python/commit/6d837ecf4d219943af065b5ac281ef08c592f596))
* preserve conversation evaluation provider configuration ([#113](https://github.com/openai/openai-guardrails-python/issues/113)) ([f20be29](https://github.com/openai/openai-guardrails-python/commit/f20be29c03f6c6cb614afee70511daa57a27f271))
* Restore type checks and enforce them in CI ([#99](https://github.com/openai/openai-guardrails-python/issues/99)) ([039500e](https://github.com/openai/openai-guardrails-python/commit/039500eeb1dd73b13bc26dc915026298bcb0e9bb))
* retain masked input in PII example history ([#110](https://github.com/openai/openai-guardrails-python/issues/110)) ([9efd83d](https://github.com/openai/openai-guardrails-python/commit/9efd83d2741ff26e53543619069e9bdb1c0181d1))
* use GITHUB_TOKEN for release automation ([#115](https://github.com/openai/openai-guardrails-python/issues/115)) ([c3e4af5](https://github.com/openai/openai-guardrails-python/commit/c3e4af5d626dd38c13d41112bad2ee1b290906f7))


### Chores

* automate releases with release-please ([#95](https://github.com/openai/openai-guardrails-python/issues/95)) ([fbae44a](https://github.com/openai/openai-guardrails-python/commit/fbae44ad1f40a91c54c8b05a23e8e9afb7820fca))
* Configure CodeQL scanning ([#94](https://github.com/openai/openai-guardrails-python/issues/94)) ([a0389d9](https://github.com/openai/openai-guardrails-python/commit/a0389d96d3c5d045c0fb4a8003db4052e9a7b6ba))
* configure Dependabot and allow authorized delivery ([#102](https://github.com/openai/openai-guardrails-python/issues/102)) ([b04de65](https://github.com/openai/openai-guardrails-python/commit/b04de65946091cea871912ae8425177af63dac70))
* restrict GitHub Actions permissions ([#96](https://github.com/openai/openai-guardrails-python/issues/96)) ([f51b6cd](https://github.com/openai/openai-guardrails-python/commit/f51b6cd8ebc67fa14179013c2dc1dfcb8cfcea65))
* Run required checks for merge queue groups ([#98](https://github.com/openai/openai-guardrails-python/issues/98)) ([dae01c7](https://github.com/openai/openai-guardrails-python/commit/dae01c79033f915b4959f0ea9ca92a9f1707a425))
* update agent assets ([7437f47](https://github.com/openai/openai-guardrails-python/commit/7437f47a7eb7ae461e159865cfa9fd6a9b77d80e))


### Documentation

* add contributor guide ([#97](https://github.com/openai/openai-guardrails-python/issues/97)) ([ec88ba7](https://github.com/openai/openai-guardrails-python/commit/ec88ba705df2ac2e0c39c6236ac7b9b4b102c49a))
* add package-specific security reporting guidance ([#101](https://github.com/openai/openai-guardrails-python/issues/101)) ([42b99ca](https://github.com/openai/openai-guardrails-python/commit/42b99ca16c802e8940d687dd457a77c55e810771))
* update readme, docs, and code comments ([741faf9](https://github.com/openai/openai-guardrails-python/commit/741faf905be05558d2a9c2681118fd59b7e5b57c))
