Ϝ
?
?

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
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ?
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
	separatorstring "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??

NoOpNoOp
?

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?	
value?	B?	 B?	
?
layer-0
layer-1
layer-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 
R

trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
 
 
 
?

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
non_trainable_variables
 
 
 
 
?

layers
layer_metrics

trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
non_trainable_variables
 
 
 
?

 layers
!layer_metrics
trainable_variables
"metrics
#layer_regularization_losses
regularization_losses
	variables
$non_trainable_variables
 
 
 
?

%layers
&layer_metrics
trainable_variables
'metrics
(layer_regularization_losses
regularization_losses
	variables
)non_trainable_variables

0
1
2
3
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
:	?*
dtype0*
shape:	?
?
PartitionedCallPartitionedCallserving_default_input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_9362
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
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
GPU 2J 8? *&
f!R
__inference__traced_save_9439
?
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_9449??
?
P
3__inference_sigmoid_act_stacked3_layer_call_fn_9341
input_2
identity?
PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_93332
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:H D

_output_shapes
:	?
!
_user_specified_name	input_2
?
F
 __inference__traced_restore_9449
file_prefix

identity_1??
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices?
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
?
c
G__inference_activation_18_layer_call_and_return_conditional_losses_9411

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	?2	
SigmoidW
IdentityIdentitySigmoid:y:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
c
G__inference_activation_16_layer_call_and_return_conditional_losses_9281

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	?2	
SigmoidW
IdentityIdentitySigmoid:y:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
O
3__inference_sigmoid_act_stacked3_layer_call_fn_9381

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_92982
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
c
G__inference_activation_18_layer_call_and_return_conditional_losses_9295

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	?2	
SigmoidW
IdentityIdentitySigmoid:y:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
P
3__inference_sigmoid_act_stacked3_layer_call_fn_9301
input_2
identity?
PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_92982
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:H D

_output_shapes
:	?
!
_user_specified_name	input_2
?
c
G__inference_activation_17_layer_call_and_return_conditional_losses_9288

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	?2	
SigmoidW
IdentityIdentitySigmoid:y:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
k
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9355
input_2
identity?
activation_16/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_16_layer_call_and_return_conditional_losses_92812
activation_16/PartitionedCall?
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_17_layer_call_and_return_conditional_losses_92882
activation_17/PartitionedCall?
activation_18/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_18_layer_call_and_return_conditional_losses_92952
activation_18/PartitionedCallr
IdentityIdentity&activation_18/PartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:H D

_output_shapes
:	?
!
_user_specified_name	input_2
?
?
"__inference_signature_wrapper_9362
input_2
identity?
PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_92692
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:H D

_output_shapes
:	?
!
_user_specified_name	input_2
?
j
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9298

inputs
identity?
activation_16/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_16_layer_call_and_return_conditional_losses_92812
activation_16/PartitionedCall?
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_17_layer_call_and_return_conditional_losses_92882
activation_17/PartitionedCall?
activation_18/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_18_layer_call_and_return_conditional_losses_92952
activation_18/PartitionedCallr
IdentityIdentity&activation_18/PartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
j
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9369

inputs
identityk
activation_16/SigmoidSigmoidinputs*
T0*
_output_shapes
:	?2
activation_16/Sigmoid~
activation_17/SigmoidSigmoidactivation_16/Sigmoid:y:0*
T0*
_output_shapes
:	?2
activation_17/Sigmoid~
activation_18/SigmoidSigmoidactivation_17/Sigmoid:y:0*
T0*
_output_shapes
:	?2
activation_18/Sigmoide
IdentityIdentityactivation_18/Sigmoid:y:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
c
G__inference_activation_17_layer_call_and_return_conditional_losses_9401

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	?2	
SigmoidW
IdentityIdentitySigmoid:y:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
j
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9376

inputs
identityk
activation_16/SigmoidSigmoidinputs*
T0*
_output_shapes
:	?2
activation_16/Sigmoid~
activation_17/SigmoidSigmoidactivation_16/Sigmoid:y:0*
T0*
_output_shapes
:	?2
activation_17/Sigmoid~
activation_18/SigmoidSigmoidactivation_17/Sigmoid:y:0*
T0*
_output_shapes
:	?2
activation_18/Sigmoide
IdentityIdentityactivation_18/Sigmoid:y:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
j
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9333

inputs
identity?
activation_16/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_16_layer_call_and_return_conditional_losses_92812
activation_16/PartitionedCall?
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_17_layer_call_and_return_conditional_losses_92882
activation_17/PartitionedCall?
activation_18/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_18_layer_call_and_return_conditional_losses_92952
activation_18/PartitionedCallr
IdentityIdentity&activation_18/PartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
H
,__inference_activation_16_layer_call_fn_9396

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_16_layer_call_and_return_conditional_losses_92812
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
c
G__inference_activation_16_layer_call_and_return_conditional_losses_9391

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	?2	
SigmoidW
IdentityIdentitySigmoid:y:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
H
,__inference_activation_18_layer_call_fn_9416

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_18_layer_call_and_return_conditional_losses_92952
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
H
,__inference_activation_17_layer_call_fn_9406

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_17_layer_call_and_return_conditional_losses_92882
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
k
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9348
input_2
identity?
activation_16/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_16_layer_call_and_return_conditional_losses_92812
activation_16/PartitionedCall?
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_17_layer_call_and_return_conditional_losses_92882
activation_17/PartitionedCall?
activation_18/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_18_layer_call_and_return_conditional_losses_92952
activation_18/PartitionedCallr
IdentityIdentity&activation_18/PartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:H D

_output_shapes
:	?
!
_user_specified_name	input_2
?
j
__inference__traced_save_9439
file_prefix
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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
?
<
__inference__wrapped_model_9269
input_2
identity?
*sigmoid_act_stacked3/activation_16/SigmoidSigmoidinput_2*
T0*
_output_shapes
:	?2,
*sigmoid_act_stacked3/activation_16/Sigmoid?
*sigmoid_act_stacked3/activation_17/SigmoidSigmoid.sigmoid_act_stacked3/activation_16/Sigmoid:y:0*
T0*
_output_shapes
:	?2,
*sigmoid_act_stacked3/activation_17/Sigmoid?
*sigmoid_act_stacked3/activation_18/SigmoidSigmoid.sigmoid_act_stacked3/activation_17/Sigmoid:y:0*
T0*
_output_shapes
:	?2,
*sigmoid_act_stacked3/activation_18/Sigmoidz
IdentityIdentity.sigmoid_act_stacked3/activation_18/Sigmoid:y:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:H D

_output_shapes
:	?
!
_user_specified_name	input_2
?
O
3__inference_sigmoid_act_stacked3_layer_call_fn_9386

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_93332
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
3
input_2(
serving_default_input_2:0	?1
activation_18 
PartitionedCall:0	?tensorflow/serving/predict:?a
?
layer-0
layer-1
layer-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
**&call_and_return_all_conditional_losses
+__call__
,_default_save_signature"?
_tf_keras_network?{"name": "sigmoid_act_stacked3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "sigmoid_act_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_16", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_17", "inbound_nodes": [[["activation_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_18", "inbound_nodes": [[["activation_17", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_18", 0, 0]]}, "shared_object_id": 4, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "sigmoid_act_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_16", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_17", "inbound_nodes": [[["activation_16", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_18", "inbound_nodes": [[["activation_17", 0, 0, {}]]], "shared_object_id": 3}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_18", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?

trainable_variables
regularization_losses
	variables
	keras_api
*-&call_and_return_all_conditional_losses
.__call__"?
_tf_keras_layer?{"name": "activation_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1}
?
trainable_variables
regularization_losses
	variables
	keras_api
*/&call_and_return_all_conditional_losses
0__call__"?
_tf_keras_layer?{"name": "activation_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["activation_16", 0, 0, {}]]], "shared_object_id": 2}
?
trainable_variables
regularization_losses
	variables
	keras_api
*1&call_and_return_all_conditional_losses
2__call__"?
_tf_keras_layer?{"name": "activation_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["activation_17", 0, 0, {}]]], "shared_object_id": 3}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
non_trainable_variables
+__call__
,_default_save_signature
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
,
3serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

layers
layer_metrics

trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
non_trainable_variables
.__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

 layers
!layer_metrics
trainable_variables
"metrics
#layer_regularization_losses
regularization_losses
	variables
$non_trainable_variables
0__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

%layers
&layer_metrics
trainable_variables
'metrics
(layer_regularization_losses
regularization_losses
	variables
)non_trainable_variables
2__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
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
?2?
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9369
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9376
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9348
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9355?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_sigmoid_act_stacked3_layer_call_fn_9301
3__inference_sigmoid_act_stacked3_layer_call_fn_9381
3__inference_sigmoid_act_stacked3_layer_call_fn_9386
3__inference_sigmoid_act_stacked3_layer_call_fn_9341?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_9269?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
?
input_2	?
?2?
G__inference_activation_16_layer_call_and_return_conditional_losses_9391?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_16_layer_call_fn_9396?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_17_layer_call_and_return_conditional_losses_9401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_17_layer_call_fn_9406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_18_layer_call_and_return_conditional_losses_9411?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_18_layer_call_fn_9416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_9362input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_9269a(?%
?
?
input_2	?
? "5?2
0
activation_18?
activation_18	??
G__inference_activation_16_layer_call_and_return_conditional_losses_9391H'?$
?
?
inputs	?
? "?
?
0	?
? k
,__inference_activation_16_layer_call_fn_9396;'?$
?
?
inputs	?
? "?	??
G__inference_activation_17_layer_call_and_return_conditional_losses_9401H'?$
?
?
inputs	?
? "?
?
0	?
? k
,__inference_activation_17_layer_call_fn_9406;'?$
?
?
inputs	?
? "?	??
G__inference_activation_18_layer_call_and_return_conditional_losses_9411H'?$
?
?
inputs	?
? "?
?
0	?
? k
,__inference_activation_18_layer_call_fn_9416;'?$
?
?
inputs	?
? "?	??
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9348Q0?-
&?#
?
input_2	?
p 

 
? "?
?
0	?
? ?
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9355Q0?-
&?#
?
input_2	?
p

 
? "?
?
0	?
? ?
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9369P/?,
%?"
?
inputs	?
p 

 
? "?
?
0	?
? ?
N__inference_sigmoid_act_stacked3_layer_call_and_return_conditional_losses_9376P/?,
%?"
?
inputs	?
p

 
? "?
?
0	?
? {
3__inference_sigmoid_act_stacked3_layer_call_fn_9301D0?-
&?#
?
input_2	?
p 

 
? "?	?{
3__inference_sigmoid_act_stacked3_layer_call_fn_9341D0?-
&?#
?
input_2	?
p

 
? "?	?z
3__inference_sigmoid_act_stacked3_layer_call_fn_9381C/?,
%?"
?
inputs	?
p 

 
? "?	?z
3__inference_sigmoid_act_stacked3_layer_call_fn_9386C/?,
%?"
?
inputs	?
p

 
? "?	??
"__inference_signature_wrapper_9362l3?0
? 
)?&
$
input_2?
input_2	?"5?2
0
activation_18?
activation_18	?