¢
ç
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Â

conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_35/kernel
~
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*'
_output_shapes
:*
dtype0
u
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_35/bias
n
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
Þ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B

layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
	bias

trainable_variables
regularization_losses
	variables
	keras_api

0
	1
 

0
	1
­

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
non_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_35/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_35/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
	1
 

0
	1
­

layers
layer_metrics

trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
non_trainable_variables

0
1
 
 
 
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:*
dtype0*
shape:
×
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_35/kernelconv2d_35/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_56685
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
æ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_56771
¹
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_35/kernelconv2d_35/bias*
Tin
2*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_56787Î¬
®
£
+__inference_many_conv2d_layer_call_fn_56656
input_1"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_many_conv2d_layer_call_and_return_conditional_losses_566402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
þ
´
 __inference__wrapped_model_56579
input_1O
4many_conv2d_conv2d_35_conv2d_readvariableop_resource:D
5many_conv2d_conv2d_35_biasadd_readvariableop_resource:	
identity¢,many_conv2d/conv2d_35/BiasAdd/ReadVariableOp¢+many_conv2d/conv2d_35/Conv2D/ReadVariableOpØ
+many_conv2d/conv2d_35/Conv2D/ReadVariableOpReadVariableOp4many_conv2d_conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02-
+many_conv2d/conv2d_35/Conv2D/ReadVariableOpß
many_conv2d/conv2d_35/Conv2DConv2Dinput_13many_conv2d/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2
many_conv2d/conv2d_35/Conv2DÏ
,many_conv2d/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp5many_conv2d_conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,many_conv2d/conv2d_35/BiasAdd/ReadVariableOpØ
many_conv2d/conv2d_35/BiasAddBiasAdd%many_conv2d/conv2d_35/Conv2D:output:04many_conv2d/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2
many_conv2d/conv2d_35/BiasAdd×
IdentityIdentity&many_conv2d/conv2d_35/BiasAdd:output:0-^many_conv2d/conv2d_35/BiasAdd/ReadVariableOp,^many_conv2d/conv2d_35/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 2\
,many_conv2d/conv2d_35/BiasAdd/ReadVariableOp,many_conv2d/conv2d_35/BiasAdd/ReadVariableOp2Z
+many_conv2d/conv2d_35/Conv2D/ReadVariableOp+many_conv2d/conv2d_35/Conv2D/ReadVariableOp:Q M
(
_output_shapes
:
!
_user_specified_name	input_1

©
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56705

inputsC
(conv2d_35_conv2d_readvariableop_resource:8
)conv2d_35_biasadd_readvariableop_resource:	
identity¢ conv2d_35/BiasAdd/ReadVariableOp¢conv2d_35/Conv2D/ReadVariableOp´
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_35/Conv2D/ReadVariableOpº
conv2d_35/Conv2DConv2Dinputs'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2
conv2d_35/Conv2D«
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp¨
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2
conv2d_35/BiasAdd³
IdentityIdentityconv2d_35/BiasAdd:output:0!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs


ÿ
D__inference_conv2d_35_layer_call_and_return_conditional_losses_56733

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs
®
£
+__inference_many_conv2d_layer_call_fn_56610
input_1"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_many_conv2d_layer_call_and_return_conditional_losses_566032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
§
 
)__inference_conv2d_35_layer_call_fn_56742

inputs"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_565962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
¦
Õ
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56603

inputs*
conv2d_35_56597:
conv2d_35_56599:	
identity¢!conv2d_35/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_35_56597conv2d_35_56599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_565962#
!conv2d_35/StatefulPartitionedCall¢
IdentityIdentity*conv2d_35/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
ð
Ë
__inference__traced_save_56771
file_prefix/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop
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
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¡
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*1
_input_shapes 
: ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::

_output_shapes
: 
¦
Õ
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56640

inputs*
conv2d_35_56634:
conv2d_35_56636:	
identity¢!conv2d_35/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_35_56634conv2d_35_56636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_565962#
!conv2d_35/StatefulPartitionedCall¢
IdentityIdentity*conv2d_35/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs


#__inference_signature_wrapper_56685
input_1"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_565792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
©
Ö
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56665
input_1*
conv2d_35_56659:
conv2d_35_56661:	
identity¢!conv2d_35/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_35_56659conv2d_35_56661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_565962#
!conv2d_35/StatefulPartitionedCall¢
IdentityIdentity*conv2d_35/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ÿ
D__inference_conv2d_35_layer_call_and_return_conditional_losses_56596

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs
«
¢
+__inference_many_conv2d_layer_call_fn_56723

inputs"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_many_conv2d_layer_call_and_return_conditional_losses_566402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs

ß
!__inference__traced_restore_56787
file_prefix<
!assignvariableop_conv2d_35_kernel:0
!assignvariableop_1_conv2d_35_bias:	

identity_3¢AssignVariableOp¢AssignVariableOp_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¡
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
RestoreV2/shape_and_slicesº
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_35_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_35_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_2

Identity_3IdentityIdentity_2:output:0^AssignVariableOp^AssignVariableOp_1*
T0*
_output_shapes
: 2

Identity_3"!

identity_3Identity_3:output:0*
_input_shapes
: : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

©
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56695

inputsC
(conv2d_35_conv2d_readvariableop_resource:8
)conv2d_35_biasadd_readvariableop_resource:	
identity¢ conv2d_35/BiasAdd/ReadVariableOp¢conv2d_35/Conv2D/ReadVariableOp´
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_35/Conv2D/ReadVariableOpº
conv2d_35/Conv2DConv2Dinputs'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2
conv2d_35/Conv2D«
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp¨
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2
conv2d_35/BiasAdd³
IdentityIdentityconv2d_35/BiasAdd:output:0!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs
«
¢
+__inference_many_conv2d_layer_call_fn_56714

inputs"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_many_conv2d_layer_call_and_return_conditional_losses_566032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
©
Ö
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56674
input_1*
conv2d_35_56668:
conv2d_35_56670:	
identity¢!conv2d_35/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_35_56668conv2d_35_56670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:~~*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_565962#
!conv2d_35/StatefulPartitionedCall¢
IdentityIdentity*conv2d_35/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall*
T0*'
_output_shapes
:~~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*­
serving_default
<
input_11
serving_default_input_1:0=
	conv2d_350
StatefulPartitionedCall:0~~tensorflow/serving/predict:O

layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*&call_and_return_all_conditional_losses
__call__
_default_save_signature"
_tf_keras_network÷{"name": "many_conv2d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "many_conv2d", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_35", 0, 0]]}, "shared_object_id": 4, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "many_conv2d", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_35", 0, 0]]}}}
÷"ô
_tf_keras_input_layerÔ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ý


kernel
	bias

trainable_variables
regularization_losses
	variables
	keras_api
*&call_and_return_all_conditional_losses
__call__"Ø	
_tf_keras_layer¾	{"name": "conv2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}}
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
Ê

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
non_trainable_variables
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
+:)2conv2d_35/kernel
:2conv2d_35/bias
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
­

layers
layer_metrics

trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
non_trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
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
æ2ã
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56695
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56705
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56665
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56674À
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
ú2÷
+__inference_many_conv2d_layer_call_fn_56610
+__inference_many_conv2d_layer_call_fn_56714
+__inference_many_conv2d_layer_call_fn_56723
+__inference_many_conv2d_layer_call_fn_56656À
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
ß2Ü
 __inference__wrapped_model_56579·
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
annotationsª *'¢$
"
input_1
î2ë
D__inference_conv2d_35_layer_call_and_return_conditional_losses_56733¢
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
Ó2Ð
)__inference_conv2d_35_layer_call_fn_56742¢
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
ÊBÇ
#__inference_signature_wrapper_56685input_1"
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
 
 __inference__wrapped_model_56579n	1¢.
'¢$
"
input_1
ª "5ª2
0
	conv2d_35# 
	conv2d_35~~¥
D__inference_conv2d_35_layer_call_and_return_conditional_losses_56733]	0¢-
&¢#
!
inputs
ª "%¢"

0~~
 }
)__inference_conv2d_35_layer_call_fn_56742P	0¢-
&¢#
!
inputs
ª "~~°
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56665f	9¢6
/¢,
"
input_1
p 

 
ª "%¢"

0~~
 °
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56674f	9¢6
/¢,
"
input_1
p

 
ª "%¢"

0~~
 ¯
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56695e	8¢5
.¢+
!
inputs
p 

 
ª "%¢"

0~~
 ¯
F__inference_many_conv2d_layer_call_and_return_conditional_losses_56705e	8¢5
.¢+
!
inputs
p

 
ª "%¢"

0~~
 
+__inference_many_conv2d_layer_call_fn_56610Y	9¢6
/¢,
"
input_1
p 

 
ª "~~
+__inference_many_conv2d_layer_call_fn_56656Y	9¢6
/¢,
"
input_1
p

 
ª "~~
+__inference_many_conv2d_layer_call_fn_56714X	8¢5
.¢+
!
inputs
p 

 
ª "~~
+__inference_many_conv2d_layer_call_fn_56723X	8¢5
.¢+
!
inputs
p

 
ª "~~ 
#__inference_signature_wrapper_56685y	<¢9
¢ 
2ª/
-
input_1"
input_1"5ª2
0
	conv2d_35# 
	conv2d_35~~