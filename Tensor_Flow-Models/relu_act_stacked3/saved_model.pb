י
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
"__inference_signature_wrapper_2630
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
__inference__traced_save_2707
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
 __inference__traced_restore_2717??
?
L
0__inference_relu_act_stacked3_layer_call_fn_2654

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
GPU 2J 8? *T
fORM
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_26012
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
M
0__inference_relu_act_stacked3_layer_call_fn_2609
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
GPU 2J 8? *T
fORM
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_26012
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
?
E
)__inference_activation_layer_call_fn_2664

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
GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_25492
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
G
+__inference_activation_2_layer_call_fn_2684

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
GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_25632
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
g
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2566

inputs
identity?
activation/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_25492
activation/PartitionedCall?
activation_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_25562
activation_1/PartitionedCall?
activation_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_25632
activation_2/PartitionedCallq
IdentityIdentity%activation_2/PartitionedCall:output:0*
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
g
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2644

inputs
identity\
activation/ReluReluinputs*
T0*
_output_shapes
:	?2
activation/Reluw
activation_1/ReluReluactivation/Relu:activations:0*
T0*
_output_shapes
:	?2
activation_1/Reluy
activation_2/ReluReluactivation_1/Relu:activations:0*
T0*
_output_shapes
:	?2
activation_2/Reluk
IdentityIdentityactivation_2/Relu:activations:0*
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
b
F__inference_activation_1_layer_call_and_return_conditional_losses_2669

inputs
identityF
ReluReluinputs*
T0*
_output_shapes
:	?2
Relu^
IdentityIdentityRelu:activations:0*
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
M
0__inference_relu_act_stacked3_layer_call_fn_2569
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
GPU 2J 8? *T
fORM
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_25662
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
`
D__inference_activation_layer_call_and_return_conditional_losses_2659

inputs
identityF
ReluReluinputs*
T0*
_output_shapes
:	?2
Relu^
IdentityIdentityRelu:activations:0*
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
b
F__inference_activation_2_layer_call_and_return_conditional_losses_2563

inputs
identityF
ReluReluinputs*
T0*
_output_shapes
:	?2
Relu^
IdentityIdentityRelu:activations:0*
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
g
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2637

inputs
identity\
activation/ReluReluinputs*
T0*
_output_shapes
:	?2
activation/Reluw
activation_1/ReluReluactivation/Relu:activations:0*
T0*
_output_shapes
:	?2
activation_1/Reluy
activation_2/ReluReluactivation_1/Relu:activations:0*
T0*
_output_shapes
:	?2
activation_2/Reluk
IdentityIdentityactivation_2/Relu:activations:0*
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
G
+__inference_activation_1_layer_call_fn_2674

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
GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_25562
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
g
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2601

inputs
identity?
activation/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_25492
activation/PartitionedCall?
activation_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_25562
activation_1/PartitionedCall?
activation_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_25632
activation_2/PartitionedCallq
IdentityIdentity%activation_2/PartitionedCall:output:0*
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
?
j
__inference__traced_save_2707
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
?
h
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2616
input_2
identity?
activation/PartitionedCallPartitionedCallinput_2*
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
GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_25492
activation/PartitionedCall?
activation_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_25562
activation_1/PartitionedCall?
activation_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_25632
activation_2/PartitionedCallq
IdentityIdentity%activation_2/PartitionedCall:output:0*
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
?
<
__inference__wrapped_model_2537
input_2
identity?
!relu_act_stacked3/activation/ReluReluinput_2*
T0*
_output_shapes
:	?2#
!relu_act_stacked3/activation/Relu?
#relu_act_stacked3/activation_1/ReluRelu/relu_act_stacked3/activation/Relu:activations:0*
T0*
_output_shapes
:	?2%
#relu_act_stacked3/activation_1/Relu?
#relu_act_stacked3/activation_2/ReluRelu1relu_act_stacked3/activation_1/Relu:activations:0*
T0*
_output_shapes
:	?2%
#relu_act_stacked3/activation_2/Relu}
IdentityIdentity1relu_act_stacked3/activation_2/Relu:activations:0*
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
b
F__inference_activation_2_layer_call_and_return_conditional_losses_2679

inputs
identityF
ReluReluinputs*
T0*
_output_shapes
:	?2
Relu^
IdentityIdentityRelu:activations:0*
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
?
F
 __inference__traced_restore_2717
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
?
h
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2623
input_2
identity?
activation/PartitionedCallPartitionedCallinput_2*
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
GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_25492
activation/PartitionedCall?
activation_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_25562
activation_1/PartitionedCall?
activation_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
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
GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_25632
activation_2/PartitionedCallq
IdentityIdentity%activation_2/PartitionedCall:output:0*
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
L
0__inference_relu_act_stacked3_layer_call_fn_2649

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
GPU 2J 8? *T
fORM
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_25662
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
?
?
"__inference_signature_wrapper_2630
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
__inference__wrapped_model_25372
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
`
D__inference_activation_layer_call_and_return_conditional_losses_2549

inputs
identityF
ReluReluinputs*
T0*
_output_shapes
:	?2
Relu^
IdentityIdentityRelu:activations:0*
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
b
F__inference_activation_1_layer_call_and_return_conditional_losses_2556

inputs
identityF
ReluReluinputs*
T0*
_output_shapes
:	?2
Relu^
IdentityIdentityRelu:activations:0*
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
serving_default_input_2:0	?0
activation_2 
PartitionedCall:0	?tensorflow/serving/predict:?_
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
,_default_save_signature"?
_tf_keras_network?{"name": "relu_act_stacked3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "relu_act_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_2", 0, 0]]}, "shared_object_id": 4, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "relu_act_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["activation", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["activation_1", 0, 0, {}]]], "shared_object_id": 3}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_2", 0, 0]]}}}
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
_tf_keras_layer?{"name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1}
?
trainable_variables
regularization_losses
	variables
	keras_api
*/&call_and_return_all_conditional_losses
0__call__"?
_tf_keras_layer?{"name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["activation", 0, 0, {}]]], "shared_object_id": 2}
?
trainable_variables
regularization_losses
	variables
	keras_api
*1&call_and_return_all_conditional_losses
2__call__"?
_tf_keras_layer?{"name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["activation_1", 0, 0, {}]]], "shared_object_id": 3}
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
?2?
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2637
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2644
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2616
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2623?
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
0__inference_relu_act_stacked3_layer_call_fn_2569
0__inference_relu_act_stacked3_layer_call_fn_2649
0__inference_relu_act_stacked3_layer_call_fn_2654
0__inference_relu_act_stacked3_layer_call_fn_2609?
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
__inference__wrapped_model_2537?
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
D__inference_activation_layer_call_and_return_conditional_losses_2659?
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
)__inference_activation_layer_call_fn_2664?
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
F__inference_activation_1_layer_call_and_return_conditional_losses_2669?
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
+__inference_activation_1_layer_call_fn_2674?
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
F__inference_activation_2_layer_call_and_return_conditional_losses_2679?
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
+__inference_activation_2_layer_call_fn_2684?
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
"__inference_signature_wrapper_2630input_2"?
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
__inference__wrapped_model_2537_(?%
?
?
input_2	?
? "3?0
.
activation_2?
activation_2	??
F__inference_activation_1_layer_call_and_return_conditional_losses_2669H'?$
?
?
inputs	?
? "?
?
0	?
? j
+__inference_activation_1_layer_call_fn_2674;'?$
?
?
inputs	?
? "?	??
F__inference_activation_2_layer_call_and_return_conditional_losses_2679H'?$
?
?
inputs	?
? "?
?
0	?
? j
+__inference_activation_2_layer_call_fn_2684;'?$
?
?
inputs	?
? "?	??
D__inference_activation_layer_call_and_return_conditional_losses_2659H'?$
?
?
inputs	?
? "?
?
0	?
? h
)__inference_activation_layer_call_fn_2664;'?$
?
?
inputs	?
? "?	??
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2616Q0?-
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
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2623Q0?-
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
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2637P/?,
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
K__inference_relu_act_stacked3_layer_call_and_return_conditional_losses_2644P/?,
%?"
?
inputs	?
p

 
? "?
?
0	?
? x
0__inference_relu_act_stacked3_layer_call_fn_2569D0?-
&?#
?
input_2	?
p 

 
? "?	?x
0__inference_relu_act_stacked3_layer_call_fn_2609D0?-
&?#
?
input_2	?
p

 
? "?	?w
0__inference_relu_act_stacked3_layer_call_fn_2649C/?,
%?"
?
inputs	?
p 

 
? "?	?w
0__inference_relu_act_stacked3_layer_call_fn_2654C/?,
%?"
?
inputs	?
p

 
? "?	??
"__inference_signature_wrapper_2630j3?0
? 
)?&
$
input_2?
input_2	?"3?0
.
activation_2?
activation_2	?