
ì¼
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718©ü
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:*
dtype0
z
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_9/kernel
s
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel* 
_output_shapes
:
*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
ä
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
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
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
j
serving_default_input_2Placeholder*
_output_shapes
:	*
dtype0*
shape:	

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_21565
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ô
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_21741
÷
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias*
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
!__inference__traced_restore_21769ïÔ


M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21609

inputs:
&dense_8_matmul_readvariableop_resource:
6
'dense_8_biasadd_readvariableop_resource:	:
&dense_9_matmul_readvariableop_resource:
6
'dense_9_biasadd_readvariableop_resource:	;
'dense_10_matmul_readvariableop_resource:
7
(dense_10_biasadd_readvariableop_resource:	
identity¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp§
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_8/MatMul¥
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_8/BiasAdd§
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMuldense_8/BiasAdd:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_9/MatMul¥
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_9/BiasAddª
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_10/MatMul/ReadVariableOp
dense_10/MatMulMatMuldense_9/BiasAdd:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_10/MatMul¨
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_10/BiasAddª
IdentityIdentitydense_10/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs
Á

2__inference_big_dense_stacked3_layer_call_fn_21408
input_2
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_213932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes
:	
!
_user_specified_name	input_2
ú

'__inference_dense_8_layer_call_fn_21662

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_213542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
«	
ö
B__inference_dense_9_layer_call_and_return_conditional_losses_21672

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs
ä

M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21393

inputs!
dense_8_21355:

dense_8_21357:	!
dense_9_21371:

dense_9_21373:	"
dense_10_21387:

dense_10_21389:	
identity¢ dense_10/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_21355dense_8_21357*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_213542!
dense_8/StatefulPartitionedCall¦
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_21371dense_9_21373*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_213702!
dense_9/StatefulPartitionedCall«
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_21387dense_10_21389*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_213862"
 dense_10/StatefulPartitionedCallÜ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs


#__inference_signature_wrapper_21565
input_2
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_213372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes
:	
!
_user_specified_name	input_2
ú

'__inference_dense_9_layer_call_fn_21681

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_213702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
«	
ö
B__inference_dense_8_layer_call_and_return_conditional_losses_21354

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs
¾

2__inference_big_dense_stacked3_layer_call_fn_21643

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_214762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
Á

2__inference_big_dense_stacked3_layer_call_fn_21508
input_2
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_214762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes
:	
!
_user_specified_name	input_2
Ñ
ø
!__inference__traced_restore_21769
file_prefix3
assignvariableop_dense_8_kernel:
.
assignvariableop_1_dense_8_bias:	5
!assignvariableop_2_dense_9_kernel:
.
assignvariableop_3_dense_9_bias:	6
"assignvariableop_4_dense_10_kernel:
/
 assignvariableop_5_dense_10_bias:	

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

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_10_biasIdentity_5:output:0"/device:CPU:0*
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
¾

2__inference_big_dense_stacked3_layer_call_fn_21626

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_213932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
«	
ö
B__inference_dense_9_layer_call_and_return_conditional_losses_21370

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs
ä

M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21476

inputs!
dense_8_21460:

dense_8_21462:	!
dense_9_21465:

dense_9_21467:	"
dense_10_21470:

dense_10_21472:	
identity¢ dense_10/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_21460dense_8_21462*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_213542!
dense_8/StatefulPartitionedCall¦
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_21465dense_9_21467*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_213702!
dense_9/StatefulPartitionedCall«
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_21470dense_10_21472*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_213862"
 dense_10/StatefulPartitionedCallÜ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
ç

M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21527
input_2!
dense_8_21511:

dense_8_21513:	!
dense_9_21516:

dense_9_21518:	"
dense_10_21521:

dense_10_21523:	
identity¢ dense_10/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_8_21511dense_8_21513*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_213542!
dense_8/StatefulPartitionedCall¦
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_21516dense_9_21518*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_213702!
dense_9/StatefulPartitionedCall«
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_21521dense_10_21523*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_213862"
 dense_10/StatefulPartitionedCallÜ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:H D

_output_shapes
:	
!
_user_specified_name	input_2
«	
ö
B__inference_dense_8_layer_call_and_return_conditional_losses_21653

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs
â

__inference__traced_save_21741
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop
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
SaveV2/shape_and_slices¾
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*P
_input_shapes?
=: :
::
::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
¬	
÷
C__inference_dense_10_layer_call_and_return_conditional_losses_21691

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs
ç

M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21546
input_2!
dense_8_21530:

dense_8_21532:	!
dense_9_21535:

dense_9_21537:	"
dense_10_21540:

dense_10_21542:	
identity¢ dense_10/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
dense_8/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_8_21530dense_8_21532*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_213542!
dense_8/StatefulPartitionedCall¦
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_21535dense_9_21537*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_213702!
dense_9/StatefulPartitionedCall«
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_21540dense_10_21542*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_213862"
 dense_10/StatefulPartitionedCallÜ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:H D

_output_shapes
:	
!
_user_specified_name	input_2
¬	
÷
C__inference_dense_10_layer_call_and_return_conditional_losses_21386

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs
Â%
Ç
 __inference__wrapped_model_21337
input_2M
9big_dense_stacked3_dense_8_matmul_readvariableop_resource:
I
:big_dense_stacked3_dense_8_biasadd_readvariableop_resource:	M
9big_dense_stacked3_dense_9_matmul_readvariableop_resource:
I
:big_dense_stacked3_dense_9_biasadd_readvariableop_resource:	N
:big_dense_stacked3_dense_10_matmul_readvariableop_resource:
J
;big_dense_stacked3_dense_10_biasadd_readvariableop_resource:	
identity¢2big_dense_stacked3/dense_10/BiasAdd/ReadVariableOp¢1big_dense_stacked3/dense_10/MatMul/ReadVariableOp¢1big_dense_stacked3/dense_8/BiasAdd/ReadVariableOp¢0big_dense_stacked3/dense_8/MatMul/ReadVariableOp¢1big_dense_stacked3/dense_9/BiasAdd/ReadVariableOp¢0big_dense_stacked3/dense_9/MatMul/ReadVariableOpà
0big_dense_stacked3/dense_8/MatMul/ReadVariableOpReadVariableOp9big_dense_stacked3_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0big_dense_stacked3/dense_8/MatMul/ReadVariableOp½
!big_dense_stacked3/dense_8/MatMulMatMulinput_28big_dense_stacked3/dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!big_dense_stacked3/dense_8/MatMulÞ
1big_dense_stacked3/dense_8/BiasAdd/ReadVariableOpReadVariableOp:big_dense_stacked3_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1big_dense_stacked3/dense_8/BiasAdd/ReadVariableOpå
"big_dense_stacked3/dense_8/BiasAddBiasAdd+big_dense_stacked3/dense_8/MatMul:product:09big_dense_stacked3/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2$
"big_dense_stacked3/dense_8/BiasAddà
0big_dense_stacked3/dense_9/MatMul/ReadVariableOpReadVariableOp9big_dense_stacked3_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0big_dense_stacked3/dense_9/MatMul/ReadVariableOpá
!big_dense_stacked3/dense_9/MatMulMatMul+big_dense_stacked3/dense_8/BiasAdd:output:08big_dense_stacked3/dense_9/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2#
!big_dense_stacked3/dense_9/MatMulÞ
1big_dense_stacked3/dense_9/BiasAdd/ReadVariableOpReadVariableOp:big_dense_stacked3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1big_dense_stacked3/dense_9/BiasAdd/ReadVariableOpå
"big_dense_stacked3/dense_9/BiasAddBiasAdd+big_dense_stacked3/dense_9/MatMul:product:09big_dense_stacked3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2$
"big_dense_stacked3/dense_9/BiasAddã
1big_dense_stacked3/dense_10/MatMul/ReadVariableOpReadVariableOp:big_dense_stacked3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1big_dense_stacked3/dense_10/MatMul/ReadVariableOpä
"big_dense_stacked3/dense_10/MatMulMatMul+big_dense_stacked3/dense_9/BiasAdd:output:09big_dense_stacked3/dense_10/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2$
"big_dense_stacked3/dense_10/MatMulá
2big_dense_stacked3/dense_10/BiasAdd/ReadVariableOpReadVariableOp;big_dense_stacked3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2big_dense_stacked3/dense_10/BiasAdd/ReadVariableOpé
#big_dense_stacked3/dense_10/BiasAddBiasAdd,big_dense_stacked3/dense_10/MatMul:product:0:big_dense_stacked3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#big_dense_stacked3/dense_10/BiasAdd¯
IdentityIdentity,big_dense_stacked3/dense_10/BiasAdd:output:03^big_dense_stacked3/dense_10/BiasAdd/ReadVariableOp2^big_dense_stacked3/dense_10/MatMul/ReadVariableOp2^big_dense_stacked3/dense_8/BiasAdd/ReadVariableOp1^big_dense_stacked3/dense_8/MatMul/ReadVariableOp2^big_dense_stacked3/dense_9/BiasAdd/ReadVariableOp1^big_dense_stacked3/dense_9/MatMul/ReadVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 2h
2big_dense_stacked3/dense_10/BiasAdd/ReadVariableOp2big_dense_stacked3/dense_10/BiasAdd/ReadVariableOp2f
1big_dense_stacked3/dense_10/MatMul/ReadVariableOp1big_dense_stacked3/dense_10/MatMul/ReadVariableOp2f
1big_dense_stacked3/dense_8/BiasAdd/ReadVariableOp1big_dense_stacked3/dense_8/BiasAdd/ReadVariableOp2d
0big_dense_stacked3/dense_8/MatMul/ReadVariableOp0big_dense_stacked3/dense_8/MatMul/ReadVariableOp2f
1big_dense_stacked3/dense_9/BiasAdd/ReadVariableOp1big_dense_stacked3/dense_9/BiasAdd/ReadVariableOp2d
0big_dense_stacked3/dense_9/MatMul/ReadVariableOp0big_dense_stacked3/dense_9/MatMul/ReadVariableOp:H D

_output_shapes
:	
!
_user_specified_name	input_2


M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21587

inputs:
&dense_8_matmul_readvariableop_resource:
6
'dense_8_biasadd_readvariableop_resource:	:
&dense_9_matmul_readvariableop_resource:
6
'dense_9_biasadd_readvariableop_resource:	;
'dense_10_matmul_readvariableop_resource:
7
(dense_10_biasadd_readvariableop_resource:	
identity¢dense_10/BiasAdd/ReadVariableOp¢dense_10/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp§
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_8/MatMul¥
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_8/BiasAdd§
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMuldense_8/BiasAdd:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_9/MatMul¥
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_9/BiasAddª
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_10/MatMul/ReadVariableOp
dense_10/MatMulMatMuldense_9/BiasAdd:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_10/MatMul¨
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_10/BiasAddª
IdentityIdentitydense_10/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:	: : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:G C

_output_shapes
:	
 
_user_specified_nameinputs
ü

(__inference_dense_10_layer_call_fn_21700

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_213862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
3
input_2(
serving_default_input_2:0	4
dense_10(
StatefulPartitionedCall:0	tensorflow/serving/predict:¨
ù&
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
2_default_save_signature"»$
_tf_keras_network${"name": "big_dense_stacked3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "big_dense_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_10", 0, 0]]}, "shared_object_id": 10, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "big_dense_stacked3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["dense_9", 0, 0, {}]]], "shared_object_id": 9}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_10", 0, 0]]}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
û


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*3&call_and_return_all_conditional_losses
4__call__"Ö
_tf_keras_layer¼{"name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
û

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*5&call_and_return_all_conditional_losses
6__call__"Ö
_tf_keras_layer¼{"name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_8", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
ý

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*7&call_and_return_all_conditional_losses
8__call__"Ø
_tf_keras_layer¾{"name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_9", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
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
": 
2dense_8/kernel
:2dense_8/bias
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
": 
2dense_9/kernel
:2dense_9/bias
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
#:!
2dense_10/kernel
:2dense_10/bias
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
2ÿ
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21587
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21609
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21527
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21546À
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
2
2__inference_big_dense_stacked3_layer_call_fn_21408
2__inference_big_dense_stacked3_layer_call_fn_21626
2__inference_big_dense_stacked3_layer_call_fn_21643
2__inference_big_dense_stacked3_layer_call_fn_21508À
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
Ö2Ó
 __inference__wrapped_model_21337®
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
ì2é
B__inference_dense_8_layer_call_and_return_conditional_losses_21653¢
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
Ñ2Î
'__inference_dense_8_layer_call_fn_21662¢
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
ì2é
B__inference_dense_9_layer_call_and_return_conditional_losses_21672¢
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
Ñ2Î
'__inference_dense_9_layer_call_fn_21681¢
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
í2ê
C__inference_dense_10_layer_call_and_return_conditional_losses_21691¢
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
Ò2Ï
(__inference_dense_10_layer_call_fn_21700¢
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
#__inference_signature_wrapper_21565input_2"
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
 
 __inference__wrapped_model_21337_
(¢%
¢

input_2	
ª "+ª(
&
dense_10
dense_10	ª
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21527Y
0¢-
&¢#

input_2	
p 

 
ª "¢

0	
 ª
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21546Y
0¢-
&¢#

input_2	
p

 
ª "¢

0	
 ©
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21587X
/¢,
%¢"

inputs	
p 

 
ª "¢

0	
 ©
M__inference_big_dense_stacked3_layer_call_and_return_conditional_losses_21609X
/¢,
%¢"

inputs	
p

 
ª "¢

0	
 
2__inference_big_dense_stacked3_layer_call_fn_21408L
0¢-
&¢#

input_2	
p 

 
ª "	
2__inference_big_dense_stacked3_layer_call_fn_21508L
0¢-
&¢#

input_2	
p

 
ª "	
2__inference_big_dense_stacked3_layer_call_fn_21626K
/¢,
%¢"

inputs	
p 

 
ª "	
2__inference_big_dense_stacked3_layer_call_fn_21643K
/¢,
%¢"

inputs	
p

 
ª "	
C__inference_dense_10_layer_call_and_return_conditional_losses_21691L'¢$
¢

inputs	
ª "¢

0	
 k
(__inference_dense_10_layer_call_fn_21700?'¢$
¢

inputs	
ª "	
B__inference_dense_8_layer_call_and_return_conditional_losses_21653L
'¢$
¢

inputs	
ª "¢

0	
 j
'__inference_dense_8_layer_call_fn_21662?
'¢$
¢

inputs	
ª "	
B__inference_dense_9_layer_call_and_return_conditional_losses_21672L'¢$
¢

inputs	
ª "¢

0	
 j
'__inference_dense_9_layer_call_fn_21681?'¢$
¢

inputs	
ª "	
#__inference_signature_wrapper_21565j
3¢0
¢ 
)ª&
$
input_2
input_2	"+ª(
&
dense_10
dense_10	