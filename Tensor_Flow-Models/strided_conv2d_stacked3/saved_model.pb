ä»
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ÿ

conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
:*
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
:*
dtype0

conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
:*
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:*
dtype0

conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
:*
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
:*
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
VARIABLE_VALUEconv2d_16/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_16/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
¢
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_42793
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
þ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_42969

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/bias*
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
!__inference__traced_restore_42997»ë
ô

!__inference__traced_restore_42997
file_prefix;
!assignvariableop_conv2d_16_kernel:/
!assignvariableop_1_conv2d_16_bias:=
#assignvariableop_2_conv2d_17_kernel:/
!assignvariableop_3_conv2d_17_bias:=
#assignvariableop_4_conv2d_18_kernel:/
!assignvariableop_5_conv2d_18_bias:

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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_16_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_17_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_17_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_18_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_18_biasIdentity_5:output:0"/device:CPU:0*
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


ý
D__inference_conv2d_18_layer_call_and_return_conditional_losses_42614

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs


ý
D__inference_conv2d_16_layer_call_and_return_conditional_losses_42582

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:2

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
ê
·
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42815

inputsB
(conv2d_16_conv2d_readvariableop_resource:7
)conv2d_16_biasadd_readvariableop_resource:B
(conv2d_17_conv2d_readvariableop_resource:7
)conv2d_17_biasadd_readvariableop_resource:B
(conv2d_18_conv2d_readvariableop_resource:7
)conv2d_18_biasadd_readvariableop_resource:
identity¢ conv2d_16/BiasAdd/ReadVariableOp¢conv2d_16/Conv2D/ReadVariableOp¢ conv2d_17/BiasAdd/ReadVariableOp¢conv2d_17/Conv2D/ReadVariableOp¢ conv2d_18/BiasAdd/ReadVariableOp¢conv2d_18/Conv2D/ReadVariableOp³
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_16/Conv2D/ReadVariableOp¹
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
conv2d_16/Conv2Dª
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp§
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d_16/BiasAdd³
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_17/Conv2D/ReadVariableOpÍ
conv2d_17/Conv2DConv2Dconv2d_16/BiasAdd:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
conv2d_17/Conv2Dª
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp§
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d_17/BiasAdd³
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_18/Conv2D/ReadVariableOpÍ
conv2d_18/Conv2DConv2Dconv2d_17/BiasAdd:output:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
conv2d_18/Conv2Dª
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp§
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d_18/BiasAdd¼
IdentityIdentityconv2d_18/BiasAdd:output:0!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs


)__inference_conv2d_17_layer_call_fn_42909

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_425982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs
â
»
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42621

inputs)
conv2d_16_42583:
conv2d_16_42585:)
conv2d_17_42599:
conv2d_17_42601:)
conv2d_18_42615:
conv2d_18_42617:
identity¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢!conv2d_18/StatefulPartitionedCall
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_16_42583conv2d_16_42585*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_425822#
!conv2d_16/StatefulPartitionedCall¹
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_42599conv2d_17_42601*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_425982#
!conv2d_17/StatefulPartitionedCall¹
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_42615conv2d_18_42617*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_426142#
!conv2d_18/StatefulPartitionedCallé
IdentityIdentity*conv2d_18/StatefulPartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
÷
¨
7__inference_strided_conv2d_stacked3_layer_call_fn_42871

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_427042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:2

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
ê
·
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42837

inputsB
(conv2d_16_conv2d_readvariableop_resource:7
)conv2d_16_biasadd_readvariableop_resource:B
(conv2d_17_conv2d_readvariableop_resource:7
)conv2d_17_biasadd_readvariableop_resource:B
(conv2d_18_conv2d_readvariableop_resource:7
)conv2d_18_biasadd_readvariableop_resource:
identity¢ conv2d_16/BiasAdd/ReadVariableOp¢conv2d_16/Conv2D/ReadVariableOp¢ conv2d_17/BiasAdd/ReadVariableOp¢conv2d_17/Conv2D/ReadVariableOp¢ conv2d_18/BiasAdd/ReadVariableOp¢conv2d_18/Conv2D/ReadVariableOp³
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_16/Conv2D/ReadVariableOp¹
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
conv2d_16/Conv2Dª
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp§
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d_16/BiasAdd³
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_17/Conv2D/ReadVariableOpÍ
conv2d_17/Conv2DConv2Dconv2d_16/BiasAdd:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
conv2d_17/Conv2Dª
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp§
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d_17/BiasAdd³
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_18/Conv2D/ReadVariableOpÍ
conv2d_18/Conv2DConv2Dconv2d_17/BiasAdd:output:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
conv2d_18/Conv2Dª
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp§
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d_18/BiasAdd¼
IdentityIdentityconv2d_18/BiasAdd:output:0!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs
÷
¨
7__inference_strided_conv2d_stacked3_layer_call_fn_42854

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_426212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:2

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
â
»
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42704

inputs)
conv2d_16_42688:
conv2d_16_42690:)
conv2d_17_42693:
conv2d_17_42695:)
conv2d_18_42698:
conv2d_18_42700:
identity¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢!conv2d_18/StatefulPartitionedCall
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_16_42688conv2d_16_42690*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_425822#
!conv2d_16/StatefulPartitionedCall¹
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_42693conv2d_17_42695*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_425982#
!conv2d_17/StatefulPartitionedCall¹
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_42698conv2d_18_42700*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_426142#
!conv2d_18/StatefulPartitionedCallé
IdentityIdentity*conv2d_18/StatefulPartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
£

)__inference_conv2d_16_layer_call_fn_42890

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_425822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:2

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


ý
D__inference_conv2d_17_layer_call_and_return_conditional_losses_42900

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs


ý
D__inference_conv2d_16_layer_call_and_return_conditional_losses_42881

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:2

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


__inference__traced_save_42969
file_prefix/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*_
_input_shapesN
L: ::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
å
¼
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42755
input_1)
conv2d_16_42739:
conv2d_16_42741:)
conv2d_17_42744:
conv2d_17_42746:)
conv2d_18_42749:
conv2d_18_42751:
identity¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢!conv2d_18/StatefulPartitionedCall
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_16_42739conv2d_16_42741*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_425822#
!conv2d_16/StatefulPartitionedCall¹
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_42744conv2d_17_42746*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_425982#
!conv2d_17/StatefulPartitionedCall¹
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_42749conv2d_18_42751*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_426142#
!conv2d_18/StatefulPartitionedCallé
IdentityIdentity*conv2d_18/StatefulPartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


)__inference_conv2d_18_layer_call_fn_42928

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_426142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs
å
¼
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42774
input_1)
conv2d_16_42758:
conv2d_16_42760:)
conv2d_17_42763:
conv2d_17_42765:)
conv2d_18_42768:
conv2d_18_42770:
identity¢!conv2d_16/StatefulPartitionedCall¢!conv2d_17/StatefulPartitionedCall¢!conv2d_18/StatefulPartitionedCall
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_16_42758conv2d_16_42760*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_425822#
!conv2d_16/StatefulPartitionedCall¹
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_42763conv2d_17_42765*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_425982#
!conv2d_17/StatefulPartitionedCall¹
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_42768conv2d_18_42770*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_426142#
!conv2d_18/StatefulPartitionedCallé
IdentityIdentity*conv2d_18/StatefulPartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1
´

#__inference_signature_wrapper_42793
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_425652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:2

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


ý
D__inference_conv2d_18_layer_call_and_return_conditional_losses_42919

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
û*
¦
 __inference__wrapped_model_42565
input_1Z
@strided_conv2d_stacked3_conv2d_16_conv2d_readvariableop_resource:O
Astrided_conv2d_stacked3_conv2d_16_biasadd_readvariableop_resource:Z
@strided_conv2d_stacked3_conv2d_17_conv2d_readvariableop_resource:O
Astrided_conv2d_stacked3_conv2d_17_biasadd_readvariableop_resource:Z
@strided_conv2d_stacked3_conv2d_18_conv2d_readvariableop_resource:O
Astrided_conv2d_stacked3_conv2d_18_biasadd_readvariableop_resource:
identity¢8strided_conv2d_stacked3/conv2d_16/BiasAdd/ReadVariableOp¢7strided_conv2d_stacked3/conv2d_16/Conv2D/ReadVariableOp¢8strided_conv2d_stacked3/conv2d_17/BiasAdd/ReadVariableOp¢7strided_conv2d_stacked3/conv2d_17/Conv2D/ReadVariableOp¢8strided_conv2d_stacked3/conv2d_18/BiasAdd/ReadVariableOp¢7strided_conv2d_stacked3/conv2d_18/Conv2D/ReadVariableOpû
7strided_conv2d_stacked3/conv2d_16/Conv2D/ReadVariableOpReadVariableOp@strided_conv2d_stacked3_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7strided_conv2d_stacked3/conv2d_16/Conv2D/ReadVariableOp
(strided_conv2d_stacked3/conv2d_16/Conv2DConv2Dinput_1?strided_conv2d_stacked3/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2*
(strided_conv2d_stacked3/conv2d_16/Conv2Dò
8strided_conv2d_stacked3/conv2d_16/BiasAdd/ReadVariableOpReadVariableOpAstrided_conv2d_stacked3_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8strided_conv2d_stacked3/conv2d_16/BiasAdd/ReadVariableOp
)strided_conv2d_stacked3/conv2d_16/BiasAddBiasAdd1strided_conv2d_stacked3/conv2d_16/Conv2D:output:0@strided_conv2d_stacked3/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)strided_conv2d_stacked3/conv2d_16/BiasAddû
7strided_conv2d_stacked3/conv2d_17/Conv2D/ReadVariableOpReadVariableOp@strided_conv2d_stacked3_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7strided_conv2d_stacked3/conv2d_17/Conv2D/ReadVariableOp­
(strided_conv2d_stacked3/conv2d_17/Conv2DConv2D2strided_conv2d_stacked3/conv2d_16/BiasAdd:output:0?strided_conv2d_stacked3/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2*
(strided_conv2d_stacked3/conv2d_17/Conv2Dò
8strided_conv2d_stacked3/conv2d_17/BiasAdd/ReadVariableOpReadVariableOpAstrided_conv2d_stacked3_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8strided_conv2d_stacked3/conv2d_17/BiasAdd/ReadVariableOp
)strided_conv2d_stacked3/conv2d_17/BiasAddBiasAdd1strided_conv2d_stacked3/conv2d_17/Conv2D:output:0@strided_conv2d_stacked3/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)strided_conv2d_stacked3/conv2d_17/BiasAddû
7strided_conv2d_stacked3/conv2d_18/Conv2D/ReadVariableOpReadVariableOp@strided_conv2d_stacked3_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7strided_conv2d_stacked3/conv2d_18/Conv2D/ReadVariableOp­
(strided_conv2d_stacked3/conv2d_18/Conv2DConv2D2strided_conv2d_stacked3/conv2d_17/BiasAdd:output:0?strided_conv2d_stacked3/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2*
(strided_conv2d_stacked3/conv2d_18/Conv2Dò
8strided_conv2d_stacked3/conv2d_18/BiasAdd/ReadVariableOpReadVariableOpAstrided_conv2d_stacked3_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8strided_conv2d_stacked3/conv2d_18/BiasAdd/ReadVariableOp
)strided_conv2d_stacked3/conv2d_18/BiasAddBiasAdd1strided_conv2d_stacked3/conv2d_18/Conv2D:output:0@strided_conv2d_stacked3/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)strided_conv2d_stacked3/conv2d_18/BiasAddä
IdentityIdentity2strided_conv2d_stacked3/conv2d_18/BiasAdd:output:09^strided_conv2d_stacked3/conv2d_16/BiasAdd/ReadVariableOp8^strided_conv2d_stacked3/conv2d_16/Conv2D/ReadVariableOp9^strided_conv2d_stacked3/conv2d_17/BiasAdd/ReadVariableOp8^strided_conv2d_stacked3/conv2d_17/Conv2D/ReadVariableOp9^strided_conv2d_stacked3/conv2d_18/BiasAdd/ReadVariableOp8^strided_conv2d_stacked3/conv2d_18/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :: : : : : : 2t
8strided_conv2d_stacked3/conv2d_16/BiasAdd/ReadVariableOp8strided_conv2d_stacked3/conv2d_16/BiasAdd/ReadVariableOp2r
7strided_conv2d_stacked3/conv2d_16/Conv2D/ReadVariableOp7strided_conv2d_stacked3/conv2d_16/Conv2D/ReadVariableOp2t
8strided_conv2d_stacked3/conv2d_17/BiasAdd/ReadVariableOp8strided_conv2d_stacked3/conv2d_17/BiasAdd/ReadVariableOp2r
7strided_conv2d_stacked3/conv2d_17/Conv2D/ReadVariableOp7strided_conv2d_stacked3/conv2d_17/Conv2D/ReadVariableOp2t
8strided_conv2d_stacked3/conv2d_18/BiasAdd/ReadVariableOp8strided_conv2d_stacked3/conv2d_18/BiasAdd/ReadVariableOp2r
7strided_conv2d_stacked3/conv2d_18/Conv2D/ReadVariableOp7strided_conv2d_stacked3/conv2d_18/Conv2D/ReadVariableOp:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ý
D__inference_conv2d_17_layer_call_and_return_conditional_losses_42598

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
ú
©
7__inference_strided_conv2d_stacked3_layer_call_fn_42736
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_427042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:2

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
ú
©
7__inference_strided_conv2d_stacked3_layer_call_fn_42636
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_426212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:2

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
_user_specified_name	input_1"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0<
	conv2d_18/
StatefulPartitionedCall:0tensorflow/serving/predict:å
3
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
2_default_save_signature"à0
_tf_keras_networkÄ0{"name": "strided_conv2d_stacked3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "strided_conv2d_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_18", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_18", 0, 0]]}, "shared_object_id": 10, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "strided_conv2d_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_18", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]], "shared_object_id": 9}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_18", 0, 0]]}}}
÷"ô
_tf_keras_input_layerÔ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ý



kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*3&call_and_return_all_conditional_losses
4__call__"Ø	
_tf_keras_layer¾	{"name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}}
ÿ


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*5&call_and_return_all_conditional_losses
6__call__"Ú	
_tf_keras_layerÀ	{"name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_16", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 26, 26, 12]}}
ý


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*7&call_and_return_all_conditional_losses
8__call__"Ø	
_tf_keras_layer¾	{"name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [5, 5]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_17", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 5, 5, 12]}}
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
*:(2conv2d_16/kernel
:2conv2d_16/bias
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
*:(2conv2d_17/kernel
:2conv2d_17/bias
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
*:(2conv2d_18/kernel
:2conv2d_18/bias
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
2
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42815
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42837
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42755
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42774À
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
ª2§
7__inference_strided_conv2d_stacked3_layer_call_fn_42636
7__inference_strided_conv2d_stacked3_layer_call_fn_42854
7__inference_strided_conv2d_stacked3_layer_call_fn_42871
7__inference_strided_conv2d_stacked3_layer_call_fn_42736À
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
 __inference__wrapped_model_42565·
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
D__inference_conv2d_16_layer_call_and_return_conditional_losses_42881¢
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
)__inference_conv2d_16_layer_call_fn_42890¢
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
D__inference_conv2d_17_layer_call_and_return_conditional_losses_42900¢
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
)__inference_conv2d_17_layer_call_fn_42909¢
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
D__inference_conv2d_18_layer_call_and_return_conditional_losses_42919¢
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
)__inference_conv2d_18_layer_call_fn_42928¢
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
#__inference_signature_wrapper_42793input_1"
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
 
 __inference__wrapped_model_42565q
1¢.
'¢$
"
input_1
ª "4ª1
/
	conv2d_18"
	conv2d_18¤
D__inference_conv2d_16_layer_call_and_return_conditional_losses_42881\
0¢-
&¢#
!
inputs
ª "$¢!

0
 |
)__inference_conv2d_16_layer_call_fn_42890O
0¢-
&¢#
!
inputs
ª "¢
D__inference_conv2d_17_layer_call_and_return_conditional_losses_42900Z.¢+
$¢!

inputs
ª "$¢!

0
 z
)__inference_conv2d_17_layer_call_fn_42909M.¢+
$¢!

inputs
ª "¢
D__inference_conv2d_18_layer_call_and_return_conditional_losses_42919Z.¢+
$¢!

inputs
ª "$¢!

0
 z
)__inference_conv2d_18_layer_call_fn_42928M.¢+
$¢!

inputs
ª "£
#__inference_signature_wrapper_42793|
<¢9
¢ 
2ª/
-
input_1"
input_1"4ª1
/
	conv2d_18"
	conv2d_18¿
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42755i
9¢6
/¢,
"
input_1
p 

 
ª "$¢!

0
 ¿
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42774i
9¢6
/¢,
"
input_1
p

 
ª "$¢!

0
 ¾
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42815h
8¢5
.¢+
!
inputs
p 

 
ª "$¢!

0
 ¾
R__inference_strided_conv2d_stacked3_layer_call_and_return_conditional_losses_42837h
8¢5
.¢+
!
inputs
p

 
ª "$¢!

0
 
7__inference_strided_conv2d_stacked3_layer_call_fn_42636\
9¢6
/¢,
"
input_1
p 

 
ª "
7__inference_strided_conv2d_stacked3_layer_call_fn_42736\
9¢6
/¢,
"
input_1
p

 
ª "
7__inference_strided_conv2d_stacked3_layer_call_fn_42854[
8¢5
.¢+
!
inputs
p 

 
ª "
7__inference_strided_conv2d_stacked3_layer_call_fn_42871[
8¢5
.¢+
!
inputs
p

 
ª "