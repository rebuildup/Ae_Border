# Release-driven solo development profile

- Status: Current default
- Constitutional authority: none; this profile must refine the Constitution
- Related: ADR-0004, ADR-0008, ADR-0012, ADR-0013, ADR-0016, ADR-0018, ADR-0022, ADR-0023

## Purpose

Ae_Border の current release-driven development で使用する Operating Model を定義します。

このprofileはproject-initの唯一の正しいorganization topologyではありません。別profileまたはproject-specific modelがConstitutionを満たす場合は置換できます。

## Current topology

### Source and implementation state

- Git: source stateのcanonical authority
- GitHub Issues: durable implementation scope / acceptance criteria / dependency state
- GitHub Pull Requests: review / candidate integration / validation evidence
- `main`: released/integrated source state
- `release-x-y-z`: current release integration line

### Release planning

- GitHub Issues を durable implementation / dependency SoT とする
- external planning surface は明示的に採用された場合だけ canonical とする

### Delivery defaults

- normal sprint cadence: 1 week
- production/stable release intent: major bump default
- normal sprint release intent: minor bump default
- within-sprint / post-introduction adjustment: patch bump default
- top-level durable work: GitHub Issue
- ticket branch: Issue number only
- independent ticket PR: target release branch
- hard dependency stack: immediate predecessor branchをbaseにできる
- first meaningful durable commit後はcanonical remote publication + Draft PRを行う
- release branchにmeaningful differenceが入ったらDraft release PRを維持する
- `main`へのnormal integrationはcurrent release branchからのrelease PRだけ
- PR landing method: merge commit only
- repository merge settings: `allow_merge_commit=true` / `allow_squash_merge=false` / `allow_rebase_merge=false`
- squash merge / rebase mergeは使用しない。branch-local `git rebase` はstack maintenance等のbranch mechanicsとして別扱い
- stacked/native landingはmerge commit semanticsを保持できる場合だけ使用する
- merge/landingはADR-0012のexplicit authorization boundaryを維持する

### Workspace/runtime defaults

- project-local toolchain/bootstrap frontend: mise
- runtime / development CLI prerequisitesは原則repository-controlledな `mise.toml` からmaterializeする
- fresh clone / CI / agentのbootstrapは `mise install`、committed mise lockfileをreproducibility mechanismとして使う場合は `mise install --locked`、command executionは `mise exec -- ...` / `mise run <task>` を標準入口とする
- required toolはexact pin、bounded version request + committed mise lockfile、または既存ecosystem-native canonical version sourceのいずれかでreproducibleにする
- `rust-toolchain.toml` 等のnative canonical version sourceがある場合、mise側へ独立した競合pinを作らない
- external/untrusted PR checkoutでrepository-controlledなmise config/taskをagentが実行する前に、documented trust reviewまたはbounded sandboxを必須とし、mise自体をisolation boundaryとして扱わない
- miseはOS/system package、container/sandbox、secret management、worker isolationの代替にしない
- mise unavailable/incompatible時は同等のversion/reproducibility guaranteeを持つ明示的fallbackを使用する
- WSL/Linux worktree frontend: Worktrunk
- Worktrunk unavailable/incompatible時: native Git worktree fallback
- worktree自体をruntime isolation proofとして扱わない
- implementation workerのmutable runtimeは適切に分離する
- parent/child handoffはimmutable identityへpinする

### Secret / environment defaults

- application / serviceのsecret valueはInfisicalをcurrent default SoTとする
- current default control planeはself-hosted `https://secrets.rebuildup.dev` とし、APIは `https://secrets.rebuildup.dev/api`
- current defaultで使用するprovider endpoint / project ID / environment/path mapping等のnon-secret pointerをrepositoryからdiscoverableにする
- wrapper / CI / runtimeはprovider endpointを明示し、managed Infisical Cloudへ暗黙fallbackしない
- required key / type / validation / non-secret metadataはrepository-controlled schemaとして保持する
- normal local workflowはCLI-firstとし、runtime injectionを優先する
- plaintext `.env` をcanonical storeにしない
- human / agent / CIはidentityを分離し、project / environment / path / actionをleast privilegeにする
- GitHub Actionsは利用可能ならOIDC + scoped Machine Identityを使用し、long-lived master credentialをdefaultにしない
- generic CIへproduction secretを渡さない
- NixOS / dotfiles / bootstrap等 encrypted secret-in-Git がexplicit requirementならSOPS等へdeviateできる
- provider-specific procedureは `secrets-management` Skillへprogressive disclosureする

### Quality / evidence

- project-specific adaptive quality profileをcompileする
- validation evidenceをcurrent artifact/SHAへbindする
- worker / integration / release gateの責務を分ける
- fixed universal required status check名は前提にしない

## Constitutional mapping

### Identity Integrity

- resolved immutable SHAへsnapshot/result/validationをpin
- predecessor change後にaffected validationを再実行
- execution generation/fencingでstale continuationを区別

### Authority Integrity

- PR readinessとmerge authorizationを分離
- merge method固定やrepository settingの整合からauthorizationを導出しない
- release/product/irreversible decisionは定義済みauthority boundaryへ従う

### Evidence Integrity

- current SHAのvalidation evidenceを使用
- false greenを禁止
- PR/release readinessはapplicable evidenceへ基づく

### Mutable Ownership Safety

- worktree-only isolationを認めない
- concurrent implementation workerはmutable runtimeを安全に分離
- DB/Redis/queue/port等のshared mutable stateを別途調停

### Organizational Continuity

- Issue / PR / Git ref / committed docs / immutable resultをdurable recovery sourceにする
- native conversation resumeをcanonical recovery stateにしない

### Canonical Consistency

- source / implementation / review / release-planning responsibilityをfield ownershipで分離
- 同一factの二重canonical化を避ける

### Progress

- 自明なimplementation decisionを不要にoperatorへ返さない
- review/validation完了後はauthorization等の実blockerがなければ次stateへ進める
- safety mechanismがdelivery deadlockを作る場合はOperating Modelをre-evaluateする

## Deviation

このprofileの具体的なtool/cadence/topologyから外れる場合でも、該当するConstitutional guaranteeを維持できれば許容します。

explicit user/project decisionとして固定されたrelease scope/version/public contract等は、単なるdefaultとは区別します。

## Re-evaluate / remove

次の場合はprofile全体または一部を見直します。

- agent/runtimeが同等以上のisolation/recovery/review semanticsをより単純に提供する
- GitHub/Linear/Worktrunk/miseの役割を別systemが置換する
- release branch modelがcontinuous delivery等に対して逆効果になる
- weekly cadenceがproject objectiveへ合わない
- eval/実績から特定procedureの追加価値が消えた


## Project-specific reconciliation

- After Effects SDK / Visual Studio / Xcode / host runtime は project-specific external environment であり、mise が置換する対象ではない。
- source-only build evidence と AE host verification evidence を分離し、tested binary / host version / candidate identity を結びつける。
- SDK checkout や host installation の mutable state を repository-owned state とみなさない。

## Agent Skills lifecycle

- current `rebuildup/project-init` を `bunx skills` で project-local に追跡する。
- canonical layout: `.agents/skills/` + `.claude/skills/`。
- bootstrap: `mise run skills-bootstrap`; update: `mise run skills-update` / `bunx skills update -p -y`。
- `skills-lock.json` は durable source/content metadata。
- `--global` は freshness evidence にしない。
