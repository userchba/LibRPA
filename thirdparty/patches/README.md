# Third-party patches

Paths are relative to each upstream archive's top-level directory; apply with `-p1`.
Each patch retains the upstream license. Upstream submission status is not recorded.

| Patch | Purpose |
| --- | --- |
| LibRI/0001-current-ddla-api.patch | Use the public GPU capability guard and handle-selected LibDDLA variable-batch calls. |
| LibComm/0001-wait-for-receive-task.patch | Wait explicitly for the receive task, which avoids the unused-result warning of `std::async`. |
| ELPA/0001-macos-thread-affinity.patch | Avoid Linux CPU-affinity APIs on macOS. |
| greenX/0001-librpa-build-integration.patch | Select LibRPA components, use relative install paths, and preserve compiler-specific flags. |

cereal uses unmodified upstream headers. LibDDLA uses unmodified upstream sources.
