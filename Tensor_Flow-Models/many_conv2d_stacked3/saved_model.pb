û»
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718
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

conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_36/kernel

$conv2d_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_36/kernel*(
_output_shapes
:*
dtype0
u
conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_36/bias
n
"conv2d_36/bias/Read/ReadVariableOpReadVariableOpconv2d_36/bias*
_output_shapes	
:*
dtype0

conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_37/kernel

$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*(
_output_shapes
:*
dtype0
u
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_37/bias
n
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
î
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*©
valueB B
ä
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 
h


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*

0
1
2
3
4
5
 
*

0
1
2
3
4
5
­

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
 non_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_35/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_35/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
­

!layers
"layer_metrics
trainable_variables
#metrics
$layer_regularization_losses
regularization_losses
	variables
%non_trainable_variables
\Z
VARIABLE_VALUEconv2d_36/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_36/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

&layers
'layer_metrics
trainable_variables
(metrics
)layer_regularization_losses
regularization_losses
	variables
*non_trainable_variables
\Z
VARIABLE_VALUEconv2d_37/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_37/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

+layers
,layer_metrics
trainable_variables
-metrics
.layer_regularization_losses
regularization_losses
	variables
/non_trainable_variables

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
|
serving_default_input_1Placeholder*(
_output_shapes
:*
dtype0*
shape:
£
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_57617
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
þ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOpConst*
Tin

2*
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
__inference__traced_save_57793

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/bias*
Tin
	2*
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
!__inference__traced_restore_57821Êë
þ
®
4__inference_many_conv2d_stacked3_layer_call_fn_57460
input_1"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_574452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
û
­
4__inference_many_conv2d_stacked3_layer_call_fn_57695

inputs"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_575282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
ë
À
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57528

inputs*
conv2d_35_57512:
conv2d_35_57514:	+
conv2d_36_57517:
conv2d_36_57519:	+
conv2d_37_57522:
conv2d_37_57524:	
identity¢!conv2d_35/StatefulPartitionedCall¢!conv2d_36/StatefulPartitionedCall¢!conv2d_37/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_35_57512conv2d_35_57514*
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
D__inference_conv2d_35_layer_call_and_return_conditional_losses_574062#
!conv2d_35/StatefulPartitionedCallº
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_57517conv2d_36_57519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_574222#
!conv2d_36/StatefulPartitionedCallº
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_57522conv2d_37_57524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_574382#
!conv2d_37/StatefulPartitionedCallê
IdentityIdentity*conv2d_37/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
î
Á
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57579
input_1*
conv2d_35_57563:
conv2d_35_57565:	+
conv2d_36_57568:
conv2d_36_57570:	+
conv2d_37_57573:
conv2d_37_57575:	
identity¢!conv2d_35/StatefulPartitionedCall¢!conv2d_36/StatefulPartitionedCall¢!conv2d_37/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_35_57563conv2d_35_57565*
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
D__inference_conv2d_35_layer_call_and_return_conditional_losses_574062#
!conv2d_35/StatefulPartitionedCallº
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_57568conv2d_36_57570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_574222#
!conv2d_36/StatefulPartitionedCallº
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_57573conv2d_37_57575*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_574382#
!conv2d_37/StatefulPartitionedCallê
IdentityIdentity*conv2d_37/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
þ
¼
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57639

inputsC
(conv2d_35_conv2d_readvariableop_resource:8
)conv2d_35_biasadd_readvariableop_resource:	D
(conv2d_36_conv2d_readvariableop_resource:8
)conv2d_36_biasadd_readvariableop_resource:	D
(conv2d_37_conv2d_readvariableop_resource:8
)conv2d_37_biasadd_readvariableop_resource:	
identity¢ conv2d_35/BiasAdd/ReadVariableOp¢conv2d_35/Conv2D/ReadVariableOp¢ conv2d_36/BiasAdd/ReadVariableOp¢conv2d_36/Conv2D/ReadVariableOp¢ conv2d_37/BiasAdd/ReadVariableOp¢conv2d_37/Conv2D/ReadVariableOp´
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
conv2d_35/BiasAddµ
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_36/Conv2D/ReadVariableOpÎ
conv2d_36/Conv2DConv2Dconv2d_35/BiasAdd:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
paddingVALID*
strides
2
conv2d_36/Conv2D«
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp¨
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:||2
conv2d_36/BiasAddµ
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_37/Conv2D/ReadVariableOpÎ
conv2d_37/Conv2DConv2Dconv2d_36/BiasAdd:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
paddingVALID*
strides
2
conv2d_37/Conv2D«
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp¨
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz2
conv2d_37/BiasAdd½
IdentityIdentityconv2d_37/BiasAdd:output:0!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs
ü

!__inference__traced_restore_57821
file_prefix<
!assignvariableop_conv2d_35_kernel:0
!assignvariableop_1_conv2d_35_bias:	?
#assignvariableop_2_conv2d_36_kernel:0
!assignvariableop_3_conv2d_36_bias:	?
#assignvariableop_4_conv2d_37_kernel:0
!assignvariableop_5_conv2d_37_bias:	

identity_7¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5ñ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ý
valueóBðB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
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
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_36_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_36_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_37_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_37_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpä

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6Ö

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¤

__inference__traced_save_57793
file_prefix/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop/
+savev2_conv2d_36_kernel_read_readvariableop-
)savev2_conv2d_36_bias_read_readvariableop/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop
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
ShardedFilenameë
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ý
valueóBðB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slicesÈ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
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

identity_1Identity_1:output:0*g
_input_shapesV
T: ::::::: 2(
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
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::

_output_shapes
: 
î
Á
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57598
input_1*
conv2d_35_57582:
conv2d_35_57584:	+
conv2d_36_57587:
conv2d_36_57589:	+
conv2d_37_57592:
conv2d_37_57594:	
identity¢!conv2d_35/StatefulPartitionedCall¢!conv2d_36/StatefulPartitionedCall¢!conv2d_37/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_35_57582conv2d_35_57584*
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
D__inference_conv2d_35_layer_call_and_return_conditional_losses_574062#
!conv2d_35/StatefulPartitionedCallº
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_57587conv2d_36_57589*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_574222#
!conv2d_36/StatefulPartitionedCallº
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_57592conv2d_37_57594*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_574382#
!conv2d_37/StatefulPartitionedCallê
IdentityIdentity*conv2d_37/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
º)

 __inference__wrapped_model_57389
input_1X
=many_conv2d_stacked3_conv2d_35_conv2d_readvariableop_resource:M
>many_conv2d_stacked3_conv2d_35_biasadd_readvariableop_resource:	Y
=many_conv2d_stacked3_conv2d_36_conv2d_readvariableop_resource:M
>many_conv2d_stacked3_conv2d_36_biasadd_readvariableop_resource:	Y
=many_conv2d_stacked3_conv2d_37_conv2d_readvariableop_resource:M
>many_conv2d_stacked3_conv2d_37_biasadd_readvariableop_resource:	
identity¢5many_conv2d_stacked3/conv2d_35/BiasAdd/ReadVariableOp¢4many_conv2d_stacked3/conv2d_35/Conv2D/ReadVariableOp¢5many_conv2d_stacked3/conv2d_36/BiasAdd/ReadVariableOp¢4many_conv2d_stacked3/conv2d_36/Conv2D/ReadVariableOp¢5many_conv2d_stacked3/conv2d_37/BiasAdd/ReadVariableOp¢4many_conv2d_stacked3/conv2d_37/Conv2D/ReadVariableOpó
4many_conv2d_stacked3/conv2d_35/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked3_conv2d_35_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype026
4many_conv2d_stacked3/conv2d_35/Conv2D/ReadVariableOpú
%many_conv2d_stacked3/conv2d_35/Conv2DConv2Dinput_1<many_conv2d_stacked3/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~*
paddingVALID*
strides
2'
%many_conv2d_stacked3/conv2d_35/Conv2Dê
5many_conv2d_stacked3/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked3_conv2d_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked3/conv2d_35/BiasAdd/ReadVariableOpü
&many_conv2d_stacked3/conv2d_35/BiasAddBiasAdd.many_conv2d_stacked3/conv2d_35/Conv2D:output:0=many_conv2d_stacked3/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:~~2(
&many_conv2d_stacked3/conv2d_35/BiasAddô
4many_conv2d_stacked3/conv2d_36/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked3_conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype026
4many_conv2d_stacked3/conv2d_36/Conv2D/ReadVariableOp¢
%many_conv2d_stacked3/conv2d_36/Conv2DConv2D/many_conv2d_stacked3/conv2d_35/BiasAdd:output:0<many_conv2d_stacked3/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
paddingVALID*
strides
2'
%many_conv2d_stacked3/conv2d_36/Conv2Dê
5many_conv2d_stacked3/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked3_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked3/conv2d_36/BiasAdd/ReadVariableOpü
&many_conv2d_stacked3/conv2d_36/BiasAddBiasAdd.many_conv2d_stacked3/conv2d_36/Conv2D:output:0=many_conv2d_stacked3/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:||2(
&many_conv2d_stacked3/conv2d_36/BiasAddô
4many_conv2d_stacked3/conv2d_37/Conv2D/ReadVariableOpReadVariableOp=many_conv2d_stacked3_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype026
4many_conv2d_stacked3/conv2d_37/Conv2D/ReadVariableOp¢
%many_conv2d_stacked3/conv2d_37/Conv2DConv2D/many_conv2d_stacked3/conv2d_36/BiasAdd:output:0<many_conv2d_stacked3/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
paddingVALID*
strides
2'
%many_conv2d_stacked3/conv2d_37/Conv2Dê
5many_conv2d_stacked3/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp>many_conv2d_stacked3_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype027
5many_conv2d_stacked3/conv2d_37/BiasAdd/ReadVariableOpü
&many_conv2d_stacked3/conv2d_37/BiasAddBiasAdd.many_conv2d_stacked3/conv2d_37/Conv2D:output:0=many_conv2d_stacked3/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz2(
&many_conv2d_stacked3/conv2d_37/BiasAddÐ
IdentityIdentity/many_conv2d_stacked3/conv2d_37/BiasAdd:output:06^many_conv2d_stacked3/conv2d_35/BiasAdd/ReadVariableOp5^many_conv2d_stacked3/conv2d_35/Conv2D/ReadVariableOp6^many_conv2d_stacked3/conv2d_36/BiasAdd/ReadVariableOp5^many_conv2d_stacked3/conv2d_36/Conv2D/ReadVariableOp6^many_conv2d_stacked3/conv2d_37/BiasAdd/ReadVariableOp5^many_conv2d_stacked3/conv2d_37/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2n
5many_conv2d_stacked3/conv2d_35/BiasAdd/ReadVariableOp5many_conv2d_stacked3/conv2d_35/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked3/conv2d_35/Conv2D/ReadVariableOp4many_conv2d_stacked3/conv2d_35/Conv2D/ReadVariableOp2n
5many_conv2d_stacked3/conv2d_36/BiasAdd/ReadVariableOp5many_conv2d_stacked3/conv2d_36/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked3/conv2d_36/Conv2D/ReadVariableOp4many_conv2d_stacked3/conv2d_36/Conv2D/ReadVariableOp2n
5many_conv2d_stacked3/conv2d_37/BiasAdd/ReadVariableOp5many_conv2d_stacked3/conv2d_37/BiasAdd/ReadVariableOp2l
4many_conv2d_stacked3/conv2d_37/Conv2D/ReadVariableOp4many_conv2d_stacked3/conv2d_37/Conv2D/ReadVariableOp:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
û
­
4__inference_many_conv2d_stacked3_layer_call_fn_57678

inputs"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_574452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
§
 
)__inference_conv2d_35_layer_call_fn_57714

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
D__inference_conv2d_35_layer_call_and_return_conditional_losses_574062
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
ë
À
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57445

inputs*
conv2d_35_57407:
conv2d_35_57409:	+
conv2d_36_57423:
conv2d_36_57425:	+
conv2d_37_57439:
conv2d_37_57441:	
identity¢!conv2d_35/StatefulPartitionedCall¢!conv2d_36/StatefulPartitionedCall¢!conv2d_37/StatefulPartitionedCall
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_35_57407conv2d_35_57409*
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
D__inference_conv2d_35_layer_call_and_return_conditional_losses_574062#
!conv2d_35/StatefulPartitionedCallº
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_57423conv2d_36_57425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_574222#
!conv2d_36/StatefulPartitionedCallº
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_57439conv2d_37_57441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_574382#
!conv2d_37/StatefulPartitionedCallê
IdentityIdentity*conv2d_37/StatefulPartitionedCall:output:0"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs



D__inference_conv2d_37_layer_call_and_return_conditional_losses_57438

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
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
:zz2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:||: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:||
 
_user_specified_nameinputs
þ
®
4__inference_many_conv2d_stacked3_layer_call_fn_57560
input_1"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_575282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ÿ
D__inference_conv2d_35_layer_call_and_return_conditional_losses_57406

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
þ
¼
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57661

inputsC
(conv2d_35_conv2d_readvariableop_resource:8
)conv2d_35_biasadd_readvariableop_resource:	D
(conv2d_36_conv2d_readvariableop_resource:8
)conv2d_36_biasadd_readvariableop_resource:	D
(conv2d_37_conv2d_readvariableop_resource:8
)conv2d_37_biasadd_readvariableop_resource:	
identity¢ conv2d_35/BiasAdd/ReadVariableOp¢conv2d_35/Conv2D/ReadVariableOp¢ conv2d_36/BiasAdd/ReadVariableOp¢conv2d_36/Conv2D/ReadVariableOp¢ conv2d_37/BiasAdd/ReadVariableOp¢conv2d_37/Conv2D/ReadVariableOp´
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
conv2d_35/BiasAddµ
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_36/Conv2D/ReadVariableOpÎ
conv2d_36/Conv2DConv2Dconv2d_35/BiasAdd:output:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
paddingVALID*
strides
2
conv2d_36/Conv2D«
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp¨
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:||2
conv2d_36/BiasAddµ
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_37/Conv2D/ReadVariableOpÎ
conv2d_37/Conv2DConv2Dconv2d_36/BiasAdd:output:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
paddingVALID*
strides
2
conv2d_37/Conv2D«
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp¨
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz2
conv2d_37/BiasAdd½
IdentityIdentityconv2d_37/BiasAdd:output:0!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs



D__inference_conv2d_36_layer_call_and_return_conditional_losses_57422

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
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
:||2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:~~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:~~
 
_user_specified_nameinputs
¾

#__inference_signature_wrapper_57617
input_1"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:zz*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_573892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ÿ
D__inference_conv2d_35_layer_call_and_return_conditional_losses_57705

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



D__inference_conv2d_37_layer_call_and_return_conditional_losses_57743

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:zz*
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
:zz2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:||: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:||
 
_user_specified_nameinputs
¦
¡
)__inference_conv2d_37_layer_call_fn_57752

inputs#
unknown:
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
:zz*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_574382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:zz2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:||: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:||
 
_user_specified_nameinputs
¦
¡
)__inference_conv2d_36_layer_call_fn_57733

inputs#
unknown:
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
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_574222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:~~: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:~~
 
_user_specified_nameinputs



D__inference_conv2d_36_layer_call_and_return_conditional_losses_57724

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:||*
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
:||2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*'
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:~~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:~~
 
_user_specified_nameinputs"ÌL
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
	conv2d_370
StatefulPartitionedCall:0zztensorflow/serving/predict:Û
3
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
*0&call_and_return_all_conditional_losses
1__call__
2_default_save_signature"Ý0
_tf_keras_networkÁ0{"name": "many_conv2d_stacked3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "many_conv2d_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_36", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_37", "inbound_nodes": [[["conv2d_36", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_37", 0, 0]]}, "shared_object_id": 10, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "many_conv2d_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_36", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_37", "inbound_nodes": [[["conv2d_36", 0, 0, {}]]], "shared_object_id": 9}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_37", 0, 0]]}}}
÷"ô
_tf_keras_input_layerÔ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
þ



kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*3&call_and_return_all_conditional_losses
4__call__"Ù	
_tf_keras_layer¿	{"name": "conv2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}}


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*5&call_and_return_all_conditional_losses
6__call__"ß	
_tf_keras_layerÅ	{"name": "conv2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_35", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 126, 126, 256]}}


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*7&call_and_return_all_conditional_losses
8__call__"ß	
_tf_keras_layerÅ	{"name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_36", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 124, 124, 256]}}
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
Ê

layers
layer_metrics
trainable_variables
metrics
layer_regularization_losses
regularization_losses
	variables
 non_trainable_variables
1__call__
2_default_save_signature
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
,
9serving_default"
signature_map
+:)2conv2d_35/kernel
:2conv2d_35/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
­

!layers
"layer_metrics
trainable_variables
#metrics
$layer_regularization_losses
regularization_losses
	variables
%non_trainable_variables
4__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_36/kernel
:2conv2d_36/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

&layers
'layer_metrics
trainable_variables
(metrics
)layer_regularization_losses
regularization_losses
	variables
*non_trainable_variables
6__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_37/kernel
:2conv2d_37/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

+layers
,layer_metrics
trainable_variables
-metrics
.layer_regularization_losses
regularization_losses
	variables
/non_trainable_variables
8__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
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
2
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57639
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57661
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57579
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57598À
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
2
4__inference_many_conv2d_stacked3_layer_call_fn_57460
4__inference_many_conv2d_stacked3_layer_call_fn_57678
4__inference_many_conv2d_stacked3_layer_call_fn_57695
4__inference_many_conv2d_stacked3_layer_call_fn_57560À
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
 __inference__wrapped_model_57389·
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
D__inference_conv2d_35_layer_call_and_return_conditional_losses_57705¢
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
)__inference_conv2d_35_layer_call_fn_57714¢
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
î2ë
D__inference_conv2d_36_layer_call_and_return_conditional_losses_57724¢
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
)__inference_conv2d_36_layer_call_fn_57733¢
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
î2ë
D__inference_conv2d_37_layer_call_and_return_conditional_losses_57743¢
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
)__inference_conv2d_37_layer_call_fn_57752¢
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
#__inference_signature_wrapper_57617input_1"
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
 
 __inference__wrapped_model_57389r
1¢.
'¢$
"
input_1
ª "5ª2
0
	conv2d_37# 
	conv2d_37zz¥
D__inference_conv2d_35_layer_call_and_return_conditional_losses_57705]
0¢-
&¢#
!
inputs
ª "%¢"

0~~
 }
)__inference_conv2d_35_layer_call_fn_57714P
0¢-
&¢#
!
inputs
ª "~~¤
D__inference_conv2d_36_layer_call_and_return_conditional_losses_57724\/¢,
%¢"
 
inputs~~
ª "%¢"

0||
 |
)__inference_conv2d_36_layer_call_fn_57733O/¢,
%¢"
 
inputs~~
ª "||¤
D__inference_conv2d_37_layer_call_and_return_conditional_losses_57743\/¢,
%¢"
 
inputs||
ª "%¢"

0zz
 |
)__inference_conv2d_37_layer_call_fn_57752O/¢,
%¢"
 
inputs||
ª "zz½
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57579j
9¢6
/¢,
"
input_1
p 

 
ª "%¢"

0zz
 ½
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57598j
9¢6
/¢,
"
input_1
p

 
ª "%¢"

0zz
 ¼
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57639i
8¢5
.¢+
!
inputs
p 

 
ª "%¢"

0zz
 ¼
O__inference_many_conv2d_stacked3_layer_call_and_return_conditional_losses_57661i
8¢5
.¢+
!
inputs
p

 
ª "%¢"

0zz
 
4__inference_many_conv2d_stacked3_layer_call_fn_57460]
9¢6
/¢,
"
input_1
p 

 
ª "zz
4__inference_many_conv2d_stacked3_layer_call_fn_57560]
9¢6
/¢,
"
input_1
p

 
ª "zz
4__inference_many_conv2d_stacked3_layer_call_fn_57678\
8¢5
.¢+
!
inputs
p 

 
ª "zz
4__inference_many_conv2d_stacked3_layer_call_fn_57695\
8¢5
.¢+
!
inputs
p

 
ª "zz¤
#__inference_signature_wrapper_57617}
<¢9
¢ 
2ª/
-
input_1"
input_1"5ª2
0
	conv2d_37# 
	conv2d_37zz