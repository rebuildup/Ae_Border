# Ae_Border Constitution

この文書は、Ae_Border の開発組織における最上位 contract です。

具体的なagent、provider、IDE、Git hosting、planning tool、worktree manager、branch命名、sprint cadence、commandはConstitutionではありません。それらは current operating model / practice が、このConstitutionを実現するために選ぶ交換可能な手段です。

## Policy hierarchy

1. **Constitution** — 組織として常に維持する性質
2. **Operating Model** — 現在の役割・planning・delivery・review・release構造
3. **Practice** — Operating Modelを実現する現在のtool / workflow
4. **Skill** — 判断が必要な場面のprogressive disclosure / playbook

下位層は上位層をrefineします。下位層の都合で上位の意味を変更しません。

## Constitutional properties

### Identity Integrity

Task、execution attempt、artifact、evidence、decision、authorityのidentityを暗黙に置換しません。

あるartifact/attemptへbindされたresultやevidenceを、別identityのcurrent result/evidenceとして扱うには、明示的で妥当な導出関係が必要です。

### Authority Integrity

重要なdecisionは、そのdecisionについてauthorityを持つactor/processだけが確定できます。

技術的に実行可能、validation済み、review済み、Readyであることは、それ自体ではauthorityを生成しません。

### Evidence Integrity

組織が主張する内容は、そのscopeと強さに対応するevidenceを持ちます。

stale / partial / unrelated / unverifiable evidenceをcurrent proofとして扱いません。

### Mutable Ownership Safety

concurrent workは、同じmutable stateへ無調停で競合するownershipを作りません。

isolation、serialization、transaction、conflict-free semantics等、同等以上のguaranteeを持つ任意のmechanismで実現できます。

### Organizational Continuity

一つのephemeral actor、conversation、process、runtime、providerの喪失だけで、重要なunfinished workを安全に理解・継続するためのorganizational stateを失わないようにします。

### Canonical Consistency

同じfactについて、reconciliation ruleのない複数のconflicting canonical authorityを意図的に維持しません。

複数systemを使う場合はfield/responsibility ownershipを分けます。

### Progress

safetyのために通常workを永久停止させません。

必要なinput/authorityが利用可能でvalid blockerがないworkは、明示的なterminal stateまたは正当なwaiting stateへ進める必要があります。

## Optimization principle

Ae_Border の開発組織の目的は current procedure への服従ではありません。

概念上、actorは次を行います。

```text
maximize project utility
subject to:
  constitutional properties
  explicit product / organizational decisions
```

project utilityの具体的なtrade-offはproject evidence、goal、risk、cost、maintainability、quality、delivery contextから判断します。

## Refinement

Operating Model / Practice は、上位propertyに対して次を説明できるようにします。

- implements
- assumptions
- guarantees
- evidence
- known limits
- deviation conditions
- re-evaluate / remove conditions

current defaultと異なる方法でも、適用される上位guaranteeを同等以上に維持するなら採用できます。

## Deviation

defaultからのdeviationは異常系ではありません。

少なくとも次を満たす場合、より良いalternativeを選択できます。

1. 適用される上位obligationが特定されている
2. alternativeが同等以上のguaranteeを維持する
3. explicit decisionを無断で上書きしない
4. materialなrisk/unknownを隠さない
5. consequential boundary変更ならdurable evidenceを残す

## Formalization boundary

Constitutionの一部はformal modelで検査します。

formal modelの成立は、実際のtool/agent/workflow implementationが自動的に正しいことを意味しません。abstract model correctness と implementation conformance は別のevidence obligationです。

## Change discipline

Constitutionへの追加は、Operating Model / Practice / Skillでは表現できない長寿命なorganizational failureを防ぐ場合に限定します。

non-constitutional ruleは、能力向上・代替mechanism・eval evidenceによって不要になった場合、意図的に降格・削除できます。

## Upstream source

`rebuildup/project-init@release-0-3-0` is the governance reconciliation baseline. Agent Skills are not frozen to that revision.
