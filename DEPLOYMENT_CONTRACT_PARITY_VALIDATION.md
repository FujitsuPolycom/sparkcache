# Deployment-contract real-inspection parity validation

Status: **qualified** for behavior preservation across the deployment-contract
extraction.

## Scope

On 2026-08-21, the extracted deployment-contract revision
`a4bfcf689f6164b2339a107d519f914fc6e7e4a6` was compared with the monolithic
profile-adapter revision
`76287b3c6d6909f4884462fe17d144e222ca3155`. The gate compared complete
transformed Docker inspection objects and complete generated Docker command
lists. Canonical JSON used sorted object keys and compact separators.

The gate was read-only on every remote host. It did not create, stop, restart,
or modify a container, route, interface, or cache entry.

## Four-Spark discovery

SparkRing's read-only `scripts/ring_doctor.py` inspected four reachable DGX
Sparks. Discovery reported:

- one valid cycle across subnets `192.168.101.0/24`, `192.168.102.0/24`,
  `192.168.103.0/24`, and `192.168.200.0/24`;
- both ring interfaces on every rank up at MTU 9000;
- forwarding and required `DOCKER-USER` relay rules present;
- `discovery_sufficient=true`, `findings=[]`, and `apply.executed=false`.

The four running containers `deepseek0731-sparkcache-r0` through
`deepseek0731-sparkcache-r3` used image ID prefix `50036224411e` and remained
up throughout the gate.

## Model-specific inspection transformations

Four DeepSeek-V4-Flash-0731 TP4/DCP1 cache-disabled source inspections and
four fixed-MTP4 GLM-5.2 EXL3 3.5-bpw TP4/DCP4 cache-disabled source inspections
(SparkRing recipe identifier `R7`) were transformed by both revisions. Every
complete transformed inspection matched byte-for-byte after canonical JSON
encoding.

| Model profile | Rank | Encoded bytes | Canonical SHA-256 |
| --- | ---: | ---: | --- |
| DeepSeek-V4-Flash-0731 TP4/DCP1 | 0 | 18,582 | `24a2051a15eb3eecc78f24f7eb6a78a25948fb6d307ef398a1a7a51da87db0f2` |
| DeepSeek-V4-Flash-0731 TP4/DCP1 | 1 | 18,616 | `5cbf7ef67a496751c2c547bc77676642b3a17e4d59be5b84d0f43795fc8570a7` |
| DeepSeek-V4-Flash-0731 TP4/DCP1 | 2 | 18,616 | `03661459214b48730c156f739b2e44241a14d2ca2344fc8ccb4fed3ed34b5fa2` |
| DeepSeek-V4-Flash-0731 TP4/DCP1 | 3 | 18,538 | `f59935af6a2fd90333ddf3cd826eeb4b805118a10f877287b28480ce1b77f911` |
| GLM-5.2 EXL3 R7 3.5-bpw TP4/DCP4 | 0 | 30,587 | `c8d971b761baf55f864536c6251a7ff49e6fc086b8eee23247624865c75841da` |
| GLM-5.2 EXL3 R7 3.5-bpw TP4/DCP4 | 1 | 30,613 | `f7badac99210eba2cb70086ed29078c93f7014802f002cb9f0d66a2eb6e92809` |
| GLM-5.2 EXL3 R7 3.5-bpw TP4/DCP4 | 2 | 30,613 | `00ca59779096cebbc3404047c24ea326d2bc9e48456cd3b804f617fe09136639` |
| GLM-5.2 EXL3 R7 3.5-bpw TP4/DCP4 | 3 | 30,613 | `6d9dd1e56a3f39697f13dca8392d7032d5a9454d2c07ddb1af6eb00d6d32033c` |

## Live-inspection Docker command construction

Each live inspection was fetched once and supplied unchanged to both
revisions. Docker execution was intercepted locally. All six complete command
lists matched exactly.

| Appliance | Rank | Command elements | Canonical SHA-256 |
| --- | ---: | ---: | --- |
| Four-Spark DeepSeek-V4-Flash-0731 TP4/DCP1 | 0 | 438 | `a4979b88e66f4599cb98925fdc86a475ce13f951cd19ba081165862b9e359926` |
| Four-Spark DeepSeek-V4-Flash-0731 TP4/DCP1 | 1 | 435 | `278b8be31339e57ee0abff82d3e57e132c88f301baa8aab3e9c12c33cd107225` |
| Four-Spark DeepSeek-V4-Flash-0731 TP4/DCP1 | 2 | 435 | `750d1fc670fc1990838e14ee4b1151d378b0a5eb937a5e5a4d2958fce225fbb0` |
| Four-Spark DeepSeek-V4-Flash-0731 TP4/DCP1 | 3 | 435 | `11d302ad4410a2a0fd063de8f27c95aa2c0fc85e174177c4f470788a71f3b5a2` |
| Two-Spark DeepSeek-V4-Flash-0731 TP2/DCP1 | 0 | 475 | `ac22c6c25246930d2a41b47ccdd8bee1450c1d1fec86545e6292c00e252f7127` |
| Two-Spark DeepSeek-V4-Flash-0731 TP2/DCP1 | 1 | 476 | `d9fefff416766c41473b44b55e278f6f82ae878e1eb111a34439aee034e4c74f` |

The two-Spark service used its existing LMCache transfer configuration; this
part of the gate qualifies model-neutral Docker command preservation, not a
SparkCache TP2 profile transformation.

## Health and conclusion

After the comparisons, both API heads returned HTTP 200 from `/health`.
`/v1/models` reported `deepseek-v4-flash-0731` with a 524,288-token limit on
the four-Spark service and `dsv4-flash` plus `leg3` with a 131,072-token limit
on the two-Spark service.

The deployment-contract extraction preserves the complete model-specific
inspection transformations and generic Docker command construction exercised
by these fourteen real inspection cases. This result does not replace model
store/restart/restore qualification or qualify a changed serving image.
