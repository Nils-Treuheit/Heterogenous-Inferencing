°
¿


8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718²¿

NoOpNoOp
Ö
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bı
×
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
 regularization_losses
!	variables
"	keras_api
R
#trainable_variables
$regularization_losses
%	variables
&	keras_api
R
'trainable_variables
(regularization_losses
)	variables
*	keras_api
R
+trainable_variables
,regularization_losses
-	variables
.	keras_api
 
 
 
­

/layers
0layer_metrics

trainable_variables
1metrics
2layer_regularization_losses
regularization_losses
	variables
3non_trainable_variables
 
 
 
 
­

4layers
5layer_metrics
trainable_variables
6metrics
7layer_regularization_losses
regularization_losses
	variables
8non_trainable_variables
 
 
 
­

9layers
:layer_metrics
trainable_variables
;metrics
<layer_regularization_losses
regularization_losses
	variables
=non_trainable_variables
 
 
 
­

>layers
?layer_metrics
trainable_variables
@metrics
Alayer_regularization_losses
regularization_losses
	variables
Bnon_trainable_variables
 
 
 
­

Clayers
Dlayer_metrics
trainable_variables
Emetrics
Flayer_regularization_losses
regularization_losses
	variables
Gnon_trainable_variables
 
 
 
­

Hlayers
Ilayer_metrics
trainable_variables
Jmetrics
Klayer_regularization_losses
 regularization_losses
!	variables
Lnon_trainable_variables
 
 
 
­

Mlayers
Nlayer_metrics
#trainable_variables
Ometrics
Player_regularization_losses
$regularization_losses
%	variables
Qnon_trainable_variables
 
 
 
­

Rlayers
Slayer_metrics
'trainable_variables
Tmetrics
Ulayer_regularization_losses
(regularization_losses
)	variables
Vnon_trainable_variables
 
 
 
­

Wlayers
Xlayer_metrics
+trainable_variables
Ymetrics
Zlayer_regularization_losses
,regularization_losses
-	variables
[non_trainable_variables
?
0
1
2
3
4
5
6
7
	8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
j
serving_default_input_2Placeholder*
_output_shapes
:	*
dtype0*
shape:	

PartitionedCallPartitionedCallserving_default_input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_7858
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_7995

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_8005¹¡
¢
H
,__inference_activation_12_layer_call_fn_7942

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_12_layer_call_and_return_conditional_losses_77252
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
ó
?
"__inference_signature_wrapper_7858
input_2
identity
PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_76852
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:H D

_output_shapes
:	
!
_user_specified_name	input_2
µ
c
G__inference_activation_13_layer_call_and_return_conditional_losses_7732

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
µ
c
G__inference_activation_11_layer_call_and_return_conditional_losses_7718

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
­
M
0__inference_tanh_act_stacked8_layer_call_fn_7752
input_2
identityÂ
PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_77492
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:H D

_output_shapes
:	
!
_user_specified_name	input_2
Ç
g
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7819

inputs
identityÖ
activation_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_76972
activation_8/PartitionedCallõ
activation_9/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_77042
activation_9/PartitionedCallø
activation_10/PartitionedCallPartitionedCall%activation_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_77112
activation_10/PartitionedCallù
activation_11/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_77182
activation_11/PartitionedCallù
activation_12/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_12_layer_call_and_return_conditional_losses_77252
activation_12/PartitionedCallù
activation_13/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_13_layer_call_and_return_conditional_losses_77322
activation_13/PartitionedCallù
activation_14/PartitionedCallPartitionedCall&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_14_layer_call_and_return_conditional_losses_77392
activation_14/PartitionedCallù
activation_15/PartitionedCallPartitionedCall&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_15_layer_call_and_return_conditional_losses_77462
activation_15/PartitionedCallr
IdentityIdentity&activation_15/PartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
µ
c
G__inference_activation_15_layer_call_and_return_conditional_losses_7967

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
µ
c
G__inference_activation_11_layer_call_and_return_conditional_losses_7927

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
µ
c
G__inference_activation_10_layer_call_and_return_conditional_losses_7917

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
¯
F
 __inference__traced_restore_8005
file_prefix

identity_1¤
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices°
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
22
	RestoreV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpd
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ê
h
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7839
input_2
identity×
activation_8/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_76972
activation_8/PartitionedCallõ
activation_9/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_77042
activation_9/PartitionedCallø
activation_10/PartitionedCallPartitionedCall%activation_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_77112
activation_10/PartitionedCallù
activation_11/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_77182
activation_11/PartitionedCallù
activation_12/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_12_layer_call_and_return_conditional_losses_77252
activation_12/PartitionedCallù
activation_13/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_13_layer_call_and_return_conditional_losses_77322
activation_13/PartitionedCallù
activation_14/PartitionedCallPartitionedCall&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_14_layer_call_and_return_conditional_losses_77392
activation_14/PartitionedCallù
activation_15/PartitionedCallPartitionedCall&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_15_layer_call_and_return_conditional_losses_77462
activation_15/PartitionedCallr
IdentityIdentity&activation_15/PartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:H D

_output_shapes
:	
!
_user_specified_name	input_2
Ó
j
__inference__traced_save_7995
file_prefix
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slicesº
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
µ
c
G__inference_activation_12_layer_call_and_return_conditional_losses_7725

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Ê
h
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7851
input_2
identity×
activation_8/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_76972
activation_8/PartitionedCallõ
activation_9/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_77042
activation_9/PartitionedCallø
activation_10/PartitionedCallPartitionedCall%activation_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_77112
activation_10/PartitionedCallù
activation_11/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_77182
activation_11/PartitionedCallù
activation_12/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_12_layer_call_and_return_conditional_losses_77252
activation_12/PartitionedCallù
activation_13/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_13_layer_call_and_return_conditional_losses_77322
activation_13/PartitionedCallù
activation_14/PartitionedCallPartitionedCall&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_14_layer_call_and_return_conditional_losses_77392
activation_14/PartitionedCallù
activation_15/PartitionedCallPartitionedCall&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_15_layer_call_and_return_conditional_losses_77462
activation_15/PartitionedCallr
IdentityIdentity&activation_15/PartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:H D

_output_shapes
:	
!
_user_specified_name	input_2
µ
c
G__inference_activation_15_layer_call_and_return_conditional_losses_7746

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
 
G
+__inference_activation_8_layer_call_fn_7902

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_76972
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
ª
L
0__inference_tanh_act_stacked8_layer_call_fn_7892

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_78192
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs

<
__inference__wrapped_model_7685
input_2
identity
#tanh_act_stacked8/activation_8/TanhTanhinput_2*
T0*
_output_shapes
:	2%
#tanh_act_stacked8/activation_8/Tanh¥
#tanh_act_stacked8/activation_9/TanhTanh'tanh_act_stacked8/activation_8/Tanh:y:0*
T0*
_output_shapes
:	2%
#tanh_act_stacked8/activation_9/Tanh§
$tanh_act_stacked8/activation_10/TanhTanh'tanh_act_stacked8/activation_9/Tanh:y:0*
T0*
_output_shapes
:	2&
$tanh_act_stacked8/activation_10/Tanh¨
$tanh_act_stacked8/activation_11/TanhTanh(tanh_act_stacked8/activation_10/Tanh:y:0*
T0*
_output_shapes
:	2&
$tanh_act_stacked8/activation_11/Tanh¨
$tanh_act_stacked8/activation_12/TanhTanh(tanh_act_stacked8/activation_11/Tanh:y:0*
T0*
_output_shapes
:	2&
$tanh_act_stacked8/activation_12/Tanh¨
$tanh_act_stacked8/activation_13/TanhTanh(tanh_act_stacked8/activation_12/Tanh:y:0*
T0*
_output_shapes
:	2&
$tanh_act_stacked8/activation_13/Tanh¨
$tanh_act_stacked8/activation_14/TanhTanh(tanh_act_stacked8/activation_13/Tanh:y:0*
T0*
_output_shapes
:	2&
$tanh_act_stacked8/activation_14/Tanh¨
$tanh_act_stacked8/activation_15/TanhTanh(tanh_act_stacked8/activation_14/Tanh:y:0*
T0*
_output_shapes
:	2&
$tanh_act_stacked8/activation_15/Tanht
IdentityIdentity(tanh_act_stacked8/activation_15/Tanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:H D

_output_shapes
:	
!
_user_specified_name	input_2
µ
c
G__inference_activation_13_layer_call_and_return_conditional_losses_7947

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
ª
L
0__inference_tanh_act_stacked8_layer_call_fn_7887

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_77492
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
µ
c
G__inference_activation_12_layer_call_and_return_conditional_losses_7937

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
¢
H
,__inference_activation_13_layer_call_fn_7952

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_13_layer_call_and_return_conditional_losses_77322
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
´
b
F__inference_activation_8_layer_call_and_return_conditional_losses_7697

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
 
G
+__inference_activation_9_layer_call_fn_7912

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_77042
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs


g
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7882

inputs
identity`
activation_8/TanhTanhinputs*
T0*
_output_shapes
:	2
activation_8/Tanho
activation_9/TanhTanhactivation_8/Tanh:y:0*
T0*
_output_shapes
:	2
activation_9/Tanhq
activation_10/TanhTanhactivation_9/Tanh:y:0*
T0*
_output_shapes
:	2
activation_10/Tanhr
activation_11/TanhTanhactivation_10/Tanh:y:0*
T0*
_output_shapes
:	2
activation_11/Tanhr
activation_12/TanhTanhactivation_11/Tanh:y:0*
T0*
_output_shapes
:	2
activation_12/Tanhr
activation_13/TanhTanhactivation_12/Tanh:y:0*
T0*
_output_shapes
:	2
activation_13/Tanhr
activation_14/TanhTanhactivation_13/Tanh:y:0*
T0*
_output_shapes
:	2
activation_14/Tanhr
activation_15/TanhTanhactivation_14/Tanh:y:0*
T0*
_output_shapes
:	2
activation_15/Tanhb
IdentityIdentityactivation_15/Tanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
¢
H
,__inference_activation_14_layer_call_fn_7962

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_14_layer_call_and_return_conditional_losses_77392
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
´
b
F__inference_activation_8_layer_call_and_return_conditional_losses_7897

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
µ
c
G__inference_activation_14_layer_call_and_return_conditional_losses_7957

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
­
M
0__inference_tanh_act_stacked8_layer_call_fn_7827
input_2
identityÂ
PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_78192
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:H D

_output_shapes
:	
!
_user_specified_name	input_2
µ
c
G__inference_activation_10_layer_call_and_return_conditional_losses_7711

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
´
b
F__inference_activation_9_layer_call_and_return_conditional_losses_7907

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
¢
H
,__inference_activation_15_layer_call_fn_7972

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_15_layer_call_and_return_conditional_losses_77462
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
µ
c
G__inference_activation_14_layer_call_and_return_conditional_losses_7739

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
¢
H
,__inference_activation_10_layer_call_fn_7922

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_77112
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs


g
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7870

inputs
identity`
activation_8/TanhTanhinputs*
T0*
_output_shapes
:	2
activation_8/Tanho
activation_9/TanhTanhactivation_8/Tanh:y:0*
T0*
_output_shapes
:	2
activation_9/Tanhq
activation_10/TanhTanhactivation_9/Tanh:y:0*
T0*
_output_shapes
:	2
activation_10/Tanhr
activation_11/TanhTanhactivation_10/Tanh:y:0*
T0*
_output_shapes
:	2
activation_11/Tanhr
activation_12/TanhTanhactivation_11/Tanh:y:0*
T0*
_output_shapes
:	2
activation_12/Tanhr
activation_13/TanhTanhactivation_12/Tanh:y:0*
T0*
_output_shapes
:	2
activation_13/Tanhr
activation_14/TanhTanhactivation_13/Tanh:y:0*
T0*
_output_shapes
:	2
activation_14/Tanhr
activation_15/TanhTanhactivation_14/Tanh:y:0*
T0*
_output_shapes
:	2
activation_15/Tanhb
IdentityIdentityactivation_15/Tanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
´
b
F__inference_activation_9_layer_call_and_return_conditional_losses_7704

inputs
identityF
TanhTanhinputs*
T0*
_output_shapes
:	2
TanhT
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Ç
g
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7749

inputs
identityÖ
activation_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_8_layer_call_and_return_conditional_losses_76972
activation_8/PartitionedCallõ
activation_9/PartitionedCallPartitionedCall%activation_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_9_layer_call_and_return_conditional_losses_77042
activation_9/PartitionedCallø
activation_10/PartitionedCallPartitionedCall%activation_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_10_layer_call_and_return_conditional_losses_77112
activation_10/PartitionedCallù
activation_11/PartitionedCallPartitionedCall&activation_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_77182
activation_11/PartitionedCallù
activation_12/PartitionedCallPartitionedCall&activation_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_12_layer_call_and_return_conditional_losses_77252
activation_12/PartitionedCallù
activation_13/PartitionedCallPartitionedCall&activation_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_13_layer_call_and_return_conditional_losses_77322
activation_13/PartitionedCallù
activation_14/PartitionedCallPartitionedCall&activation_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_14_layer_call_and_return_conditional_losses_77392
activation_14/PartitionedCallù
activation_15/PartitionedCallPartitionedCall&activation_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_15_layer_call_and_return_conditional_losses_77462
activation_15/PartitionedCallr
IdentityIdentity&activation_15/PartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
¢
H
,__inference_activation_11_layer_call_fn_7932

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_11_layer_call_and_return_conditional_losses_77182
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs"ÌJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
3
input_2(
serving_default_input_2:0	1
activation_15 
PartitionedCall:0	tensorflow/serving/predict:ì¹
ó(
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

trainable_variables
regularization_losses
	variables
	keras_api

signatures
*\&call_and_return_all_conditional_losses
]__call__
^_default_save_signature"Â&
_tf_keras_network¦&{"name": "tanh_act_stacked8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "tanh_act_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_8", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_9", "inbound_nodes": [[["activation_8", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_10", "inbound_nodes": [[["activation_9", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_11", "inbound_nodes": [[["activation_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_12", "inbound_nodes": [[["activation_11", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_13", "inbound_nodes": [[["activation_12", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_14", "inbound_nodes": [[["activation_13", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_15", "inbound_nodes": [[["activation_14", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_15", 0, 0]]}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "tanh_act_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_8", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Activation", "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_9", "inbound_nodes": [[["activation_8", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_10", "inbound_nodes": [[["activation_9", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_11", "inbound_nodes": [[["activation_10", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_12", "inbound_nodes": [[["activation_11", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_13", "inbound_nodes": [[["activation_12", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_14", "inbound_nodes": [[["activation_13", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "tanh"}, "name": "activation_15", "inbound_nodes": [[["activation_14", 0, 0, {}]]], "shared_object_id": 8}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_15", 0, 0]]}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}

trainable_variables
regularization_losses
	variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"
_tf_keras_layerï{"name": "activation_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_8", "trainable": true, "dtype": "float32", "activation": "tanh"}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1}

trainable_variables
regularization_losses
	variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"
_tf_keras_layerô{"name": "activation_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_9", "trainable": true, "dtype": "float32", "activation": "tanh"}, "inbound_nodes": [[["activation_8", 0, 0, {}]]], "shared_object_id": 2}

trainable_variables
regularization_losses
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"
_tf_keras_layerö{"name": "activation_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_10", "trainable": true, "dtype": "float32", "activation": "tanh"}, "inbound_nodes": [[["activation_9", 0, 0, {}]]], "shared_object_id": 3}
 
trainable_variables
regularization_losses
	variables
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"
_tf_keras_layer÷{"name": "activation_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_11", "trainable": true, "dtype": "float32", "activation": "tanh"}, "inbound_nodes": [[["activation_10", 0, 0, {}]]], "shared_object_id": 4}
 
trainable_variables
 regularization_losses
!	variables
"	keras_api
*g&call_and_return_all_conditional_losses
h__call__"
_tf_keras_layer÷{"name": "activation_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_12", "trainable": true, "dtype": "float32", "activation": "tanh"}, "inbound_nodes": [[["activation_11", 0, 0, {}]]], "shared_object_id": 5}
 
#trainable_variables
$regularization_losses
%	variables
&	keras_api
*i&call_and_return_all_conditional_losses
j__call__"
_tf_keras_layer÷{"name": "activation_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_13", "trainable": true, "dtype": "float32", "activation": "tanh"}, "inbound_nodes": [[["activation_12", 0, 0, {}]]], "shared_object_id": 6}
 
'trainable_variables
(regularization_losses
)	variables
*	keras_api
*k&call_and_return_all_conditional_losses
l__call__"
_tf_keras_layer÷{"name": "activation_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "tanh"}, "inbound_nodes": [[["activation_13", 0, 0, {}]]], "shared_object_id": 7}
 
+trainable_variables
,regularization_losses
-	variables
.	keras_api
*m&call_and_return_all_conditional_losses
n__call__"
_tf_keras_layer÷{"name": "activation_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "tanh"}, "inbound_nodes": [[["activation_14", 0, 0, {}]]], "shared_object_id": 8}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ê

/layers
0layer_metrics

trainable_variables
1metrics
2layer_regularization_losses
regularization_losses
	variables
3non_trainable_variables
]__call__
^_default_save_signature
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
,
oserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

4layers
5layer_metrics
trainable_variables
6metrics
7layer_regularization_losses
regularization_losses
	variables
8non_trainable_variables
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

9layers
:layer_metrics
trainable_variables
;metrics
<layer_regularization_losses
regularization_losses
	variables
=non_trainable_variables
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

>layers
?layer_metrics
trainable_variables
@metrics
Alayer_regularization_losses
regularization_losses
	variables
Bnon_trainable_variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Clayers
Dlayer_metrics
trainable_variables
Emetrics
Flayer_regularization_losses
regularization_losses
	variables
Gnon_trainable_variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Hlayers
Ilayer_metrics
trainable_variables
Jmetrics
Klayer_regularization_losses
 regularization_losses
!	variables
Lnon_trainable_variables
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Mlayers
Nlayer_metrics
#trainable_variables
Ometrics
Player_regularization_losses
$regularization_losses
%	variables
Qnon_trainable_variables
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Rlayers
Slayer_metrics
'trainable_variables
Tmetrics
Ulayer_regularization_losses
(regularization_losses
)	variables
Vnon_trainable_variables
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Wlayers
Xlayer_metrics
+trainable_variables
Ymetrics
Zlayer_regularization_losses
,regularization_losses
-	variables
[non_trainable_variables
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ú2÷
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7870
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7882
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7839
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7851À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_tanh_act_stacked8_layer_call_fn_7752
0__inference_tanh_act_stacked8_layer_call_fn_7887
0__inference_tanh_act_stacked8_layer_call_fn_7892
0__inference_tanh_act_stacked8_layer_call_fn_7827À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
__inference__wrapped_model_7685®
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢

input_2	
ğ2í
F__inference_activation_8_layer_call_and_return_conditional_losses_7897¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_activation_8_layer_call_fn_7902¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ğ2í
F__inference_activation_9_layer_call_and_return_conditional_losses_7907¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_activation_9_layer_call_fn_7912¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_10_layer_call_and_return_conditional_losses_7917¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_10_layer_call_fn_7922¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_11_layer_call_and_return_conditional_losses_7927¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_11_layer_call_fn_7932¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_12_layer_call_and_return_conditional_losses_7937¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_12_layer_call_fn_7942¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_13_layer_call_and_return_conditional_losses_7947¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_13_layer_call_fn_7952¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_14_layer_call_and_return_conditional_losses_7957¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_14_layer_call_fn_7962¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_15_layer_call_and_return_conditional_losses_7967¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_15_layer_call_fn_7972¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÉBÆ
"__inference_signature_wrapper_7858input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
__inference__wrapped_model_7685a(¢%
¢

input_2	
ª "5ª2
0
activation_15
activation_15	
G__inference_activation_10_layer_call_and_return_conditional_losses_7917H'¢$
¢

inputs	
ª "¢

0	
 k
,__inference_activation_10_layer_call_fn_7922;'¢$
¢

inputs	
ª "	
G__inference_activation_11_layer_call_and_return_conditional_losses_7927H'¢$
¢

inputs	
ª "¢

0	
 k
,__inference_activation_11_layer_call_fn_7932;'¢$
¢

inputs	
ª "	
G__inference_activation_12_layer_call_and_return_conditional_losses_7937H'¢$
¢

inputs	
ª "¢

0	
 k
,__inference_activation_12_layer_call_fn_7942;'¢$
¢

inputs	
ª "	
G__inference_activation_13_layer_call_and_return_conditional_losses_7947H'¢$
¢

inputs	
ª "¢

0	
 k
,__inference_activation_13_layer_call_fn_7952;'¢$
¢

inputs	
ª "	
G__inference_activation_14_layer_call_and_return_conditional_losses_7957H'¢$
¢

inputs	
ª "¢

0	
 k
,__inference_activation_14_layer_call_fn_7962;'¢$
¢

inputs	
ª "	
G__inference_activation_15_layer_call_and_return_conditional_losses_7967H'¢$
¢

inputs	
ª "¢

0	
 k
,__inference_activation_15_layer_call_fn_7972;'¢$
¢

inputs	
ª "	
F__inference_activation_8_layer_call_and_return_conditional_losses_7897H'¢$
¢

inputs	
ª "¢

0	
 j
+__inference_activation_8_layer_call_fn_7902;'¢$
¢

inputs	
ª "	
F__inference_activation_9_layer_call_and_return_conditional_losses_7907H'¢$
¢

inputs	
ª "¢

0	
 j
+__inference_activation_9_layer_call_fn_7912;'¢$
¢

inputs	
ª "	
"__inference_signature_wrapper_7858l3¢0
¢ 
)ª&
$
input_2
input_2	"5ª2
0
activation_15
activation_15	 
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7839Q0¢-
&¢#

input_2	
p 

 
ª "¢

0	
  
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7851Q0¢-
&¢#

input_2	
p

 
ª "¢

0	
 
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7870P/¢,
%¢"

inputs	
p 

 
ª "¢

0	
 
K__inference_tanh_act_stacked8_layer_call_and_return_conditional_losses_7882P/¢,
%¢"

inputs	
p

 
ª "¢

0	
 x
0__inference_tanh_act_stacked8_layer_call_fn_7752D0¢-
&¢#

input_2	
p 

 
ª "	x
0__inference_tanh_act_stacked8_layer_call_fn_7827D0¢-
&¢#

input_2	
p

 
ª "	w
0__inference_tanh_act_stacked8_layer_call_fn_7887C/¢,
%¢"

inputs	
p 

 
ª "	w
0__inference_tanh_act_stacked8_layer_call_fn_7892C/¢,
%¢"

inputs	
p

 
ª "	