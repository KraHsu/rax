# Reading initial instruction metadata

`services.robotwin.instruction_metadata.readonly_instruction_metadata(task, native)`
supports `blocks_ranking_size` and `place_bread_basket` at the audited native commit
`0008ae6800df9f75fc8de7098bacb01735fd8fd2`. Call immediately after `setup_demo`,
before any action. The adapter verifies the native task source SHA256, reads
existing actor poses and asset IDs, and returns `info` plus provenance. It does
not call `get_info`, `play_once`, motion helpers, or scene removal methods, and
does not write `native.info`.

`blocks_ranking_size` maps block1/2/3 to large/medium/small and their initial
left/right arm. Expert execution order is irrelevant to these identity fields.
`place_bread_basket` preserves basket/bread asset IDs and optional second-bread
`{C}`. Its arm label comes from the initial bread side(s). The original single-
bread `get_info` moves the bread before reading its side; copying that behavior
would make the instruction describe a moved object rather than the chosen arm.
Empty or mismatched bread lists, invalid IDs, nonfinite coordinates, and bread
exactly on the center line fail explicitly. Native sampling excludes that line,
where the original motion and metadata branches disagree.

Unknown tasks return `None`. This is not permission to call an arbitrary task's
`get_info`; callers must choose a separately reviewed path. OpenWAM's fixed-
horizon model collector uses the adapter for these two tasks and compares scene
fingerprints before/after. Other tasks retain its existing isolated expert run,
followed by closing and rebuilding the reference scene before model actions.
The current cached benchmark also uses separate expert eligibility scenes and
sealed instructions; it does not call these unsafe `get_info` methods.

Deploy the collector with a RAX snapshot containing this module. Do not replace
running frozen snapshots. Instruction receipts record adapter version/source
hash, task source hash, read fields, and whether expert actions were used.

Regression fixtures contain the exact audited task sources and native instruction
templates with a source receipt. CPU tests extract the native classes without
loading simulation dependencies, forbid all motion/removal entry points, check
unchanged actor poses, and verify real instruction placeholders for single- and
two-bread cases. This validates metadata initialization, not task success.
