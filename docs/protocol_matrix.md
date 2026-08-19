# Protocol matrix

| Protocol | Disjoint test students | Held-out target KC excluded from fitting/validation | Target labels select configuration | Online personal update after prediction | Primary first position |
|---|---:|---:|---:|---:|---:|
| E1 fixed-validation | yes | mixed natural coverage | validation pool | yes | Q2 |
| E3 early history | yes | mixed natural coverage | validation pool | yes | Q2--Q5 |
| E4 random unseen KC | no | yes | no | yes | first target encounter |
| E5 cluster unseen KC | no | yes | no | yes | first target encounter |
| E6 double cold start | yes | yes | no | yes | first target encounter |
| E7 online adaptation | protocol-dependent | yes | no | yes | every held-out encounter |
| E8 target-only K-shot | yes | not applicable | internal K-student validation | yes | Q2 |
| E8 source-only online | yes | target domain excluded | no | yes | Q2 unless separately stratified |
| E8 source-to-target | yes | target test excluded | internal K-student validation | yes | Q2 |
| E8 multi-source-to-target | yes | target test excluded | internal K-student validation | yes | Q2 |
| E10 semantic controls | parent condition | parent condition | no test-label selection | yes | parent condition |
| E12 vocabulary expansion | yes | yes for added concepts | no | yes | Q2 or later |
| E14 robustness replay | yes | not applicable | no | yes | original target; history perturbed |
| E15 calibration | parent condition | parent condition | validation temperature only | yes | parent condition |
| E20 system cases | yes | mixed natural coverage | validation pool | yes | Q2 |
| E20 unseen case | no | yes | no | yes | first target encounter |
| E20 double case | yes | yes | no | yes | first target encounter |

Every emitted prediction records its protocol, sequence position, seen status,
calibration status, training size, holdout ratio, and transfer metadata. The
manifest generator rejects overlapping student sets and any held-out concept
component in training or validation interactions. Identifier models use one
shared OOV row; held-out descriptors do not enter SemOpKT smoothness graphs;
GKT inserts unseen nodes only after global fitting.

Cross-domain runs may encode target descriptors because those descriptions are
the query interface, but inducing anchors, smoothness graphs, and source graphs
are initialized from source-training concepts only. The source checkpoint is
selected before any target test sequence is constructed or evaluated. For
source-only runs, target responses are accessed only after their corresponding
predictions to update that individual student's online state; they never fit,
select, or calibrate global parameters.
