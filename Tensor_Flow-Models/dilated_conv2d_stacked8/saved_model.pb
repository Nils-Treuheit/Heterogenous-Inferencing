??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
BatchToSpaceND

input"T
block_shape"Tblock_shape
crops"Tcrops
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
Tcropstype0:
2	
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
?
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
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
SpaceToBatchND

input"T
block_shape"Tblock_shape
paddings"	Tpaddings
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
	Tpaddingstype0:
2	
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
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0
?
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
:*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:*
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:*
dtype0
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:*
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?'
value?'B?' B?'
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
h

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
h

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
h

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
h

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
v
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15
 
v
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15
?

?layers
@layer_metrics

trainable_variables
Ametrics
Blayer_regularization_losses
regularization_losses
	variables
Cnon_trainable_variables
 
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Dlayers
Elayer_metrics
trainable_variables
Fmetrics
Glayer_regularization_losses
regularization_losses
	variables
Hnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Ilayers
Jlayer_metrics
trainable_variables
Kmetrics
Llayer_regularization_losses
regularization_losses
	variables
Mnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Nlayers
Olayer_metrics
trainable_variables
Pmetrics
Qlayer_regularization_losses
regularization_losses
	variables
Rnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
?

Slayers
Tlayer_metrics
#trainable_variables
Umetrics
Vlayer_regularization_losses
$regularization_losses
%	variables
Wnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
?

Xlayers
Ylayer_metrics
)trainable_variables
Zmetrics
[layer_regularization_losses
*regularization_losses
+	variables
\non_trainable_variables
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
?

]layers
^layer_metrics
/trainable_variables
_metrics
`layer_regularization_losses
0regularization_losses
1	variables
anon_trainable_variables
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
?

blayers
clayer_metrics
5trainable_variables
dmetrics
elayer_regularization_losses
6regularization_losses
7	variables
fnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
?

glayers
hlayer_metrics
;trainable_variables
imetrics
jlayer_regularization_losses
<regularization_losses
=	variables
knon_trainable_variables
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
|
serving_default_input_1Placeholder*(
_output_shapes
:??*
dtype0*
shape:??
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_36986
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_37651
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/bias*
Tin
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_37709??
?
?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_36474

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"t   t   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:::2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:88*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:pp2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:tt: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:tt
 
_user_specified_nameinputs
?
?
D__inference_conv2d_14_layer_call_and_return_conditional_losses_36555

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"h   h   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:442
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:22*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:dd2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:dd2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:dd2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:hh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:hh
 
_user_specified_nameinputs
?
?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_37391

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:xx2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:||: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:||
 
_user_specified_nameinputs
?
?
)__inference_conv2d_12_layer_call_fn_37490

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:ll*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_365012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:ll2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:pp: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:pp
 
_user_specified_nameinputs
??
?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_37266

inputsA
'conv2d_8_conv2d_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:A
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource:B
(conv2d_12_conv2d_readvariableop_resource:7
)conv2d_12_biasadd_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource:7
)conv2d_13_biasadd_readvariableop_resource:B
(conv2d_14_conv2d_readvariableop_resource:7
)conv2d_14_biasadd_readvariableop_resource:B
(conv2d_15_conv2d_readvariableop_resource:7
)conv2d_15_biasadd_readvariableop_resource:
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_8/Conv2D/dilation_rate?
<conv2d_8/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2>
<conv2d_8/Conv2D/required_space_to_batch_paddings/input_shape?
>conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2@
>conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddings?
9conv2d_8/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2;
9conv2d_8/Conv2D/required_space_to_batch_paddings/paddings?
6conv2d_8/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                28
6conv2d_8/Conv2D/required_space_to_batch_paddings/crops?
*conv2d_8/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_8/Conv2D/SpaceToBatchND/block_shape?
'conv2d_8/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2)
'conv2d_8/Conv2D/SpaceToBatchND/paddings?
conv2d_8/Conv2D/SpaceToBatchNDSpaceToBatchNDinputs3conv2d_8/Conv2D/SpaceToBatchND/block_shape:output:00conv2d_8/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@2 
conv2d_8/Conv2D/SpaceToBatchND?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2D'conv2d_8/Conv2D/SpaceToBatchND:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2
conv2d_8/Conv2D?
*conv2d_8/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_8/Conv2D/BatchToSpaceND/block_shape?
$conv2d_8/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2&
$conv2d_8/Conv2D/BatchToSpaceND/crops?
conv2d_8/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_8/Conv2D:output:03conv2d_8/Conv2D/BatchToSpaceND/block_shape:output:0-conv2d_8/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||2 
conv2d_8/Conv2D/BatchToSpaceND?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAdd'conv2d_8/Conv2D/BatchToSpaceND:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2
conv2d_8/BiasAdd?
conv2d_9/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_9/Conv2D/dilation_rate?
<conv2d_9/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   2>
<conv2d_9/Conv2D/required_space_to_batch_paddings/input_shape?
>conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2@
>conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddings?
9conv2d_9/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2;
9conv2d_9/Conv2D/required_space_to_batch_paddings/paddings?
6conv2d_9/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                28
6conv2d_9/Conv2D/required_space_to_batch_paddings/crops?
*conv2d_9/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_9/Conv2D/SpaceToBatchND/block_shape?
'conv2d_9/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2)
'conv2d_9/Conv2D/SpaceToBatchND/paddings?
conv2d_9/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_8/BiasAdd:output:03conv2d_9/Conv2D/SpaceToBatchND/block_shape:output:00conv2d_9/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>2 
conv2d_9/Conv2D/SpaceToBatchND?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2D'conv2d_9/Conv2D/SpaceToBatchND:output:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2
conv2d_9/Conv2D?
*conv2d_9/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_9/Conv2D/BatchToSpaceND/block_shape?
$conv2d_9/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2&
$conv2d_9/Conv2D/BatchToSpaceND/crops?
conv2d_9/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_9/Conv2D:output:03conv2d_9/Conv2D/BatchToSpaceND/block_shape:output:0-conv2d_9/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx2 
conv2d_9/Conv2D/BatchToSpaceND?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAdd'conv2d_9/Conv2D/BatchToSpaceND:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2
conv2d_9/BiasAdd?
conv2d_10/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_10/Conv2D/dilation_rate?
=conv2d_10/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   2?
=conv2d_10/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_10/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_10/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_10/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_10/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_10/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_10/Conv2D/SpaceToBatchND/block_shape?
(conv2d_10/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_10/Conv2D/SpaceToBatchND/paddings?
conv2d_10/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_9/BiasAdd:output:04conv2d_10/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_10/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<2!
conv2d_10/Conv2D/SpaceToBatchND?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2D(conv2d_10/Conv2D/SpaceToBatchND:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2
conv2d_10/Conv2D?
+conv2d_10/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_10/Conv2D/BatchToSpaceND/block_shape?
%conv2d_10/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_10/Conv2D/BatchToSpaceND/crops?
conv2d_10/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_10/Conv2D:output:04conv2d_10/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_10/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt2!
conv2d_10/Conv2D/BatchToSpaceND?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAdd(conv2d_10/Conv2D/BatchToSpaceND:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2
conv2d_10/BiasAdd?
conv2d_11/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_11/Conv2D/dilation_rate?
=conv2d_11/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"t   t   2?
=conv2d_11/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_11/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_11/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_11/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_11/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_11/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_11/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_11/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_11/Conv2D/SpaceToBatchND/block_shape?
(conv2d_11/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_11/Conv2D/SpaceToBatchND/paddings?
conv2d_11/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_10/BiasAdd:output:04conv2d_11/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_11/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:::2!
conv2d_11/Conv2D/SpaceToBatchND?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2D(conv2d_11/Conv2D/SpaceToBatchND:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:88*
paddingVALID*
strides
2
conv2d_11/Conv2D?
+conv2d_11/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_11/Conv2D/BatchToSpaceND/block_shape?
%conv2d_11/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_11/Conv2D/BatchToSpaceND/crops?
conv2d_11/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_11/Conv2D:output:04conv2d_11/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_11/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:pp2!
conv2d_11/Conv2D/BatchToSpaceND?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAdd(conv2d_11/Conv2D/BatchToSpaceND:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp2
conv2d_11/BiasAdd?
conv2d_12/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_12/Conv2D/dilation_rate?
=conv2d_12/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"p   p   2?
=conv2d_12/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_12/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_12/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_12/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_12/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_12/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_12/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_12/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_12/Conv2D/SpaceToBatchND/block_shape?
(conv2d_12/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_12/Conv2D/SpaceToBatchND/paddings?
conv2d_12/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_11/BiasAdd:output:04conv2d_12/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_12/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:882!
conv2d_12/Conv2D/SpaceToBatchND?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D(conv2d_12/Conv2D/SpaceToBatchND:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:66*
paddingVALID*
strides
2
conv2d_12/Conv2D?
+conv2d_12/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_12/Conv2D/BatchToSpaceND/block_shape?
%conv2d_12/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_12/Conv2D/BatchToSpaceND/crops?
conv2d_12/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_12/Conv2D:output:04conv2d_12/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_12/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:ll2!
conv2d_12/Conv2D/BatchToSpaceND?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAdd(conv2d_12/Conv2D/BatchToSpaceND:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:ll2
conv2d_12/BiasAdd?
conv2d_13/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_13/Conv2D/dilation_rate?
=conv2d_13/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"l   l   2?
=conv2d_13/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_13/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_13/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_13/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_13/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_13/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_13/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_13/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_13/Conv2D/SpaceToBatchND/block_shape?
(conv2d_13/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_13/Conv2D/SpaceToBatchND/paddings?
conv2d_13/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_12/BiasAdd:output:04conv2d_13/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_13/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:662!
conv2d_13/Conv2D/SpaceToBatchND?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D(conv2d_13/Conv2D/SpaceToBatchND:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingVALID*
strides
2
conv2d_13/Conv2D?
+conv2d_13/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_13/Conv2D/BatchToSpaceND/block_shape?
%conv2d_13/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_13/Conv2D/BatchToSpaceND/crops?
conv2d_13/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_13/Conv2D:output:04conv2d_13/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_13/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:hh2!
conv2d_13/Conv2D/BatchToSpaceND?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAdd(conv2d_13/Conv2D/BatchToSpaceND:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
conv2d_13/BiasAdd?
conv2d_14/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_14/Conv2D/dilation_rate?
=conv2d_14/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"h   h   2?
=conv2d_14/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_14/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_14/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_14/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_14/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_14/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_14/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_14/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_14/Conv2D/SpaceToBatchND/block_shape?
(conv2d_14/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_14/Conv2D/SpaceToBatchND/paddings?
conv2d_14/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_13/BiasAdd:output:04conv2d_14/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_14/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:442!
conv2d_14/Conv2D/SpaceToBatchND?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D(conv2d_14/Conv2D/SpaceToBatchND:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:22*
paddingVALID*
strides
2
conv2d_14/Conv2D?
+conv2d_14/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_14/Conv2D/BatchToSpaceND/block_shape?
%conv2d_14/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_14/Conv2D/BatchToSpaceND/crops?
conv2d_14/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_14/Conv2D:output:04conv2d_14/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_14/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:dd2!
conv2d_14/Conv2D/BatchToSpaceND?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAdd(conv2d_14/Conv2D/BatchToSpaceND:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:dd2
conv2d_14/BiasAdd?
conv2d_15/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_15/Conv2D/dilation_rate?
=conv2d_15/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   2?
=conv2d_15/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_15/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_15/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_15/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_15/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_15/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_15/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_15/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_15/Conv2D/SpaceToBatchND/block_shape?
(conv2d_15/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_15/Conv2D/SpaceToBatchND/paddings?
conv2d_15/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_14/BiasAdd:output:04conv2d_15/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_15/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:222!
conv2d_15/Conv2D/SpaceToBatchND?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2D(conv2d_15/Conv2D/SpaceToBatchND:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:00*
paddingVALID*
strides
2
conv2d_15/Conv2D?
+conv2d_15/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_15/Conv2D/BatchToSpaceND/block_shape?
%conv2d_15/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_15/Conv2D/BatchToSpaceND/crops?
conv2d_15/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_15/Conv2D:output:04conv2d_15/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_15/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:``2!
conv2d_15/Conv2D/BatchToSpaceND?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAdd(conv2d_15/Conv2D/BatchToSpaceND:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:``2
conv2d_15/BiasAdd?
IdentityIdentityconv2d_15/BiasAdd:output:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_37451

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"t   t   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:::2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:88*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:pp2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:tt: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:tt
 
_user_specified_nameinputs
?
?
)__inference_conv2d_13_layer_call_fn_37520

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_365282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:hh2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ll: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:ll
 
_user_specified_nameinputs
?
?
7__inference_dilated_conv2d_stacked8_layer_call_fn_36624
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_365892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?
?
D__inference_conv2d_13_layer_call_and_return_conditional_losses_37511

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"l   l   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:662
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:hh2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:hh2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ll: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:ll
 
_user_specified_nameinputs
??
?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_37126

inputsA
'conv2d_8_conv2d_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:A
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource:B
(conv2d_12_conv2d_readvariableop_resource:7
)conv2d_12_biasadd_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource:7
)conv2d_13_biasadd_readvariableop_resource:B
(conv2d_14_conv2d_readvariableop_resource:7
)conv2d_14_biasadd_readvariableop_resource:B
(conv2d_15_conv2d_readvariableop_resource:7
)conv2d_15_biasadd_readvariableop_resource:
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_8/Conv2D/dilation_rate?
<conv2d_8/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2>
<conv2d_8/Conv2D/required_space_to_batch_paddings/input_shape?
>conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2@
>conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddings?
9conv2d_8/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2;
9conv2d_8/Conv2D/required_space_to_batch_paddings/paddings?
6conv2d_8/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                28
6conv2d_8/Conv2D/required_space_to_batch_paddings/crops?
*conv2d_8/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_8/Conv2D/SpaceToBatchND/block_shape?
'conv2d_8/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2)
'conv2d_8/Conv2D/SpaceToBatchND/paddings?
conv2d_8/Conv2D/SpaceToBatchNDSpaceToBatchNDinputs3conv2d_8/Conv2D/SpaceToBatchND/block_shape:output:00conv2d_8/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@2 
conv2d_8/Conv2D/SpaceToBatchND?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2D'conv2d_8/Conv2D/SpaceToBatchND:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2
conv2d_8/Conv2D?
*conv2d_8/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_8/Conv2D/BatchToSpaceND/block_shape?
$conv2d_8/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2&
$conv2d_8/Conv2D/BatchToSpaceND/crops?
conv2d_8/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_8/Conv2D:output:03conv2d_8/Conv2D/BatchToSpaceND/block_shape:output:0-conv2d_8/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||2 
conv2d_8/Conv2D/BatchToSpaceND?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAdd'conv2d_8/Conv2D/BatchToSpaceND:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2
conv2d_8/BiasAdd?
conv2d_9/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
conv2d_9/Conv2D/dilation_rate?
<conv2d_9/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   2>
<conv2d_9/Conv2D/required_space_to_batch_paddings/input_shape?
>conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2@
>conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddings?
9conv2d_9/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2;
9conv2d_9/Conv2D/required_space_to_batch_paddings/paddings?
6conv2d_9/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                28
6conv2d_9/Conv2D/required_space_to_batch_paddings/crops?
*conv2d_9/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_9/Conv2D/SpaceToBatchND/block_shape?
'conv2d_9/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2)
'conv2d_9/Conv2D/SpaceToBatchND/paddings?
conv2d_9/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_8/BiasAdd:output:03conv2d_9/Conv2D/SpaceToBatchND/block_shape:output:00conv2d_9/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>2 
conv2d_9/Conv2D/SpaceToBatchND?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2D'conv2d_9/Conv2D/SpaceToBatchND:output:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2
conv2d_9/Conv2D?
*conv2d_9/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*conv2d_9/Conv2D/BatchToSpaceND/block_shape?
$conv2d_9/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2&
$conv2d_9/Conv2D/BatchToSpaceND/crops?
conv2d_9/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_9/Conv2D:output:03conv2d_9/Conv2D/BatchToSpaceND/block_shape:output:0-conv2d_9/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx2 
conv2d_9/Conv2D/BatchToSpaceND?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAdd'conv2d_9/Conv2D/BatchToSpaceND:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2
conv2d_9/BiasAdd?
conv2d_10/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_10/Conv2D/dilation_rate?
=conv2d_10/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   2?
=conv2d_10/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_10/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_10/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_10/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_10/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_10/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_10/Conv2D/SpaceToBatchND/block_shape?
(conv2d_10/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_10/Conv2D/SpaceToBatchND/paddings?
conv2d_10/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_9/BiasAdd:output:04conv2d_10/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_10/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<2!
conv2d_10/Conv2D/SpaceToBatchND?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2D(conv2d_10/Conv2D/SpaceToBatchND:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2
conv2d_10/Conv2D?
+conv2d_10/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_10/Conv2D/BatchToSpaceND/block_shape?
%conv2d_10/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_10/Conv2D/BatchToSpaceND/crops?
conv2d_10/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_10/Conv2D:output:04conv2d_10/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_10/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt2!
conv2d_10/Conv2D/BatchToSpaceND?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAdd(conv2d_10/Conv2D/BatchToSpaceND:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2
conv2d_10/BiasAdd?
conv2d_11/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_11/Conv2D/dilation_rate?
=conv2d_11/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"t   t   2?
=conv2d_11/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_11/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_11/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_11/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_11/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_11/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_11/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_11/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_11/Conv2D/SpaceToBatchND/block_shape?
(conv2d_11/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_11/Conv2D/SpaceToBatchND/paddings?
conv2d_11/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_10/BiasAdd:output:04conv2d_11/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_11/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:::2!
conv2d_11/Conv2D/SpaceToBatchND?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2D(conv2d_11/Conv2D/SpaceToBatchND:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:88*
paddingVALID*
strides
2
conv2d_11/Conv2D?
+conv2d_11/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_11/Conv2D/BatchToSpaceND/block_shape?
%conv2d_11/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_11/Conv2D/BatchToSpaceND/crops?
conv2d_11/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_11/Conv2D:output:04conv2d_11/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_11/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:pp2!
conv2d_11/Conv2D/BatchToSpaceND?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAdd(conv2d_11/Conv2D/BatchToSpaceND:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp2
conv2d_11/BiasAdd?
conv2d_12/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_12/Conv2D/dilation_rate?
=conv2d_12/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"p   p   2?
=conv2d_12/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_12/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_12/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_12/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_12/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_12/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_12/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_12/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_12/Conv2D/SpaceToBatchND/block_shape?
(conv2d_12/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_12/Conv2D/SpaceToBatchND/paddings?
conv2d_12/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_11/BiasAdd:output:04conv2d_12/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_12/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:882!
conv2d_12/Conv2D/SpaceToBatchND?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D(conv2d_12/Conv2D/SpaceToBatchND:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:66*
paddingVALID*
strides
2
conv2d_12/Conv2D?
+conv2d_12/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_12/Conv2D/BatchToSpaceND/block_shape?
%conv2d_12/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_12/Conv2D/BatchToSpaceND/crops?
conv2d_12/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_12/Conv2D:output:04conv2d_12/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_12/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:ll2!
conv2d_12/Conv2D/BatchToSpaceND?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAdd(conv2d_12/Conv2D/BatchToSpaceND:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:ll2
conv2d_12/BiasAdd?
conv2d_13/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_13/Conv2D/dilation_rate?
=conv2d_13/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"l   l   2?
=conv2d_13/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_13/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_13/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_13/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_13/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_13/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_13/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_13/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_13/Conv2D/SpaceToBatchND/block_shape?
(conv2d_13/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_13/Conv2D/SpaceToBatchND/paddings?
conv2d_13/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_12/BiasAdd:output:04conv2d_13/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_13/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:662!
conv2d_13/Conv2D/SpaceToBatchND?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2D(conv2d_13/Conv2D/SpaceToBatchND:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingVALID*
strides
2
conv2d_13/Conv2D?
+conv2d_13/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_13/Conv2D/BatchToSpaceND/block_shape?
%conv2d_13/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_13/Conv2D/BatchToSpaceND/crops?
conv2d_13/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_13/Conv2D:output:04conv2d_13/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_13/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:hh2!
conv2d_13/Conv2D/BatchToSpaceND?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAdd(conv2d_13/Conv2D/BatchToSpaceND:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2
conv2d_13/BiasAdd?
conv2d_14/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_14/Conv2D/dilation_rate?
=conv2d_14/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"h   h   2?
=conv2d_14/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_14/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_14/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_14/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_14/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_14/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_14/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_14/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_14/Conv2D/SpaceToBatchND/block_shape?
(conv2d_14/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_14/Conv2D/SpaceToBatchND/paddings?
conv2d_14/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_13/BiasAdd:output:04conv2d_14/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_14/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:442!
conv2d_14/Conv2D/SpaceToBatchND?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D(conv2d_14/Conv2D/SpaceToBatchND:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:22*
paddingVALID*
strides
2
conv2d_14/Conv2D?
+conv2d_14/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_14/Conv2D/BatchToSpaceND/block_shape?
%conv2d_14/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_14/Conv2D/BatchToSpaceND/crops?
conv2d_14/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_14/Conv2D:output:04conv2d_14/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_14/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:dd2!
conv2d_14/Conv2D/BatchToSpaceND?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAdd(conv2d_14/Conv2D/BatchToSpaceND:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:dd2
conv2d_14/BiasAdd?
conv2d_15/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
conv2d_15/Conv2D/dilation_rate?
=conv2d_15/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   2?
=conv2d_15/Conv2D/required_space_to_batch_paddings/input_shape?
?conv2d_15/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?conv2d_15/Conv2D/required_space_to_batch_paddings/base_paddings?
:conv2d_15/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2<
:conv2d_15/Conv2D/required_space_to_batch_paddings/paddings?
7conv2d_15/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                29
7conv2d_15/Conv2D/required_space_to_batch_paddings/crops?
+conv2d_15/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_15/Conv2D/SpaceToBatchND/block_shape?
(conv2d_15/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2*
(conv2d_15/Conv2D/SpaceToBatchND/paddings?
conv2d_15/Conv2D/SpaceToBatchNDSpaceToBatchNDconv2d_14/BiasAdd:output:04conv2d_15/Conv2D/SpaceToBatchND/block_shape:output:01conv2d_15/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:222!
conv2d_15/Conv2D/SpaceToBatchND?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2D(conv2d_15/Conv2D/SpaceToBatchND:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:00*
paddingVALID*
strides
2
conv2d_15/Conv2D?
+conv2d_15/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+conv2d_15/Conv2D/BatchToSpaceND/block_shape?
%conv2d_15/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2'
%conv2d_15/Conv2D/BatchToSpaceND/crops?
conv2d_15/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_15/Conv2D:output:04conv2d_15/Conv2D/BatchToSpaceND/block_shape:output:0.conv2d_15/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:``2!
conv2d_15/Conv2D/BatchToSpaceND?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAdd(conv2d_15/Conv2D/BatchToSpaceND:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:``2
conv2d_15/BiasAdd?
IdentityIdentityconv2d_15/BiasAdd:output:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?F
?

!__inference__traced_restore_37709
file_prefix:
 assignvariableop_conv2d_8_kernel:.
 assignvariableop_1_conv2d_8_bias:<
"assignvariableop_2_conv2d_9_kernel:.
 assignvariableop_3_conv2d_9_bias:=
#assignvariableop_4_conv2d_10_kernel:/
!assignvariableop_5_conv2d_10_bias:=
#assignvariableop_6_conv2d_11_kernel:/
!assignvariableop_7_conv2d_11_bias:=
#assignvariableop_8_conv2d_12_kernel:/
!assignvariableop_9_conv2d_12_bias:>
$assignvariableop_10_conv2d_13_kernel:0
"assignvariableop_11_conv2d_13_bias:>
$assignvariableop_12_conv2d_14_kernel:0
"assignvariableop_13_conv2d_14_bias:>
$assignvariableop_14_conv2d_15_kernel:0
"assignvariableop_15_conv2d_15_bias:
identity_17??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_13_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_15_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_15_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16?
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_37361

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?,
?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_36589

inputs(
conv2d_8_36394:
conv2d_8_36396:(
conv2d_9_36421:
conv2d_9_36423:)
conv2d_10_36448:
conv2d_10_36450:)
conv2d_11_36475:
conv2d_11_36477:)
conv2d_12_36502:
conv2d_12_36504:)
conv2d_13_36529:
conv2d_13_36531:)
conv2d_14_36556:
conv2d_14_36558:)
conv2d_15_36583:
conv2d_15_36585:
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_36394conv2d_8_36396*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_363932"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_36421conv2d_9_36423*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_364202"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_36448conv2d_10_36450*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_364472#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_36475conv2d_11_36477*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_364742#
!conv2d_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_36502conv2d_12_36504*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:ll*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_365012#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_36529conv2d_13_36531*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_365282#
!conv2d_13/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_36556conv2d_14_36558*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:dd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_365552#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_36583conv2d_15_36585*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_365822#
!conv2d_15/StatefulPartitionedCall?
IdentityIdentity*conv2d_15/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?,
?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_36787

inputs(
conv2d_8_36746:
conv2d_8_36748:(
conv2d_9_36751:
conv2d_9_36753:)
conv2d_10_36756:
conv2d_10_36758:)
conv2d_11_36761:
conv2d_11_36763:)
conv2d_12_36766:
conv2d_12_36768:)
conv2d_13_36771:
conv2d_13_36773:)
conv2d_14_36776:
conv2d_14_36778:)
conv2d_15_36781:
conv2d_15_36783:
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_36746conv2d_8_36748*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_363932"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_36751conv2d_9_36753*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_364202"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_36756conv2d_10_36758*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_364472#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_36761conv2d_11_36763*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_364742#
!conv2d_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_36766conv2d_12_36768*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:ll*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_365012#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_36771conv2d_13_36773*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_365282#
!conv2d_13/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_36776conv2d_14_36778*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:dd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_365552#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_36781conv2d_15_36783*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_365822#
!conv2d_15/StatefulPartitionedCall?
IdentityIdentity*conv2d_15/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
(__inference_conv2d_8_layer_call_fn_37370

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_363932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_36582

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:222
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:00*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:``2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:``2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:dd
 
_user_specified_nameinputs
?
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_36393

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:||2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
(__inference_conv2d_9_layer_call_fn_37400

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_364202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:xx2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:||: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:||
 
_user_specified_nameinputs
?
?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_36420

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:xx2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:||: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:||
 
_user_specified_nameinputs
?
?
)__inference_conv2d_10_layer_call_fn_37430

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_364472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:xx: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:xx
 
_user_specified_nameinputs
?,
?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_36947
input_1(
conv2d_8_36906:
conv2d_8_36908:(
conv2d_9_36911:
conv2d_9_36913:)
conv2d_10_36916:
conv2d_10_36918:)
conv2d_11_36921:
conv2d_11_36923:)
conv2d_12_36926:
conv2d_12_36928:)
conv2d_13_36931:
conv2d_13_36933:)
conv2d_14_36936:
conv2d_14_36938:)
conv2d_15_36941:
conv2d_15_36943:
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_8_36906conv2d_8_36908*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_363932"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_36911conv2d_9_36913*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_364202"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_36916conv2d_10_36918*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_364472#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_36921conv2d_11_36923*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_364742#
!conv2d_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_36926conv2d_12_36928*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:ll*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_365012#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_36931conv2d_13_36933*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_365282#
!conv2d_13/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_36936conv2d_14_36938*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:dd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_365552#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_36941conv2d_15_36943*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_365822#
!conv2d_15/StatefulPartitionedCall?
IdentityIdentity*conv2d_15/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?
?
7__inference_dilated_conv2d_stacked8_layer_call_fn_36859
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_367872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?
?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_37571

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:222
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:00*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:``2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:``2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:dd
 
_user_specified_nameinputs
?
?
)__inference_conv2d_14_layer_call_fn_37550

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:dd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_365552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:dd2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:hh: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:hh
 
_user_specified_nameinputs
?
?
D__inference_conv2d_14_layer_call_and_return_conditional_losses_37541

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"h   h   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:442
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:22*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:dd2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:dd2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:dd2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:hh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:hh
 
_user_specified_nameinputs
?
?
D__inference_conv2d_12_layer_call_and_return_conditional_losses_36501

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"p   p   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:882
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:66*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:ll2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:ll2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:ll2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:pp
 
_user_specified_nameinputs
?+
?
__inference__traced_save_37651
file_prefix.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::::::::: 2(
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
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?,
?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_36903
input_1(
conv2d_8_36862:
conv2d_8_36864:(
conv2d_9_36867:
conv2d_9_36869:)
conv2d_10_36872:
conv2d_10_36874:)
conv2d_11_36877:
conv2d_11_36879:)
conv2d_12_36882:
conv2d_12_36884:)
conv2d_13_36887:
conv2d_13_36889:)
conv2d_14_36892:
conv2d_14_36894:)
conv2d_15_36897:
conv2d_15_36899:
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_8_36862conv2d_8_36864*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_363932"
 conv2d_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_36867conv2d_9_36869*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_364202"
 conv2d_9/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_36872conv2d_10_36874*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_364472#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_36877conv2d_11_36879*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_364742#
!conv2d_11/StatefulPartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0conv2d_12_36882conv2d_12_36884*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:ll*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_365012#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_36887conv2d_13_36889*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:hh*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_365282#
!conv2d_13/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_36892conv2d_14_36894*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:dd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_365552#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_36897conv2d_15_36899*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_365822#
!conv2d_15/StatefulPartitionedCall?
IdentityIdentity*conv2d_15/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?
?
D__inference_conv2d_12_layer_call_and_return_conditional_losses_37481

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"p   p   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:882
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:66*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:ll2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:ll2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:ll2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:pp
 
_user_specified_nameinputs
?
?
)__inference_conv2d_15_layer_call_fn_37580

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_365822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:dd
 
_user_specified_nameinputs
?
?
D__inference_conv2d_13_layer_call_and_return_conditional_losses_36528

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"l   l   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:662
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:hh2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:hh2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ll: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:ll
 
_user_specified_nameinputs
?
?
)__inference_conv2d_11_layer_call_fn_37460

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_364742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:tt: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:tt
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_36986
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_363652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
Ŧ
?
 __inference__wrapped_model_36365
input_1Y
?dilated_conv2d_stacked8_conv2d_8_conv2d_readvariableop_resource:N
@dilated_conv2d_stacked8_conv2d_8_biasadd_readvariableop_resource:Y
?dilated_conv2d_stacked8_conv2d_9_conv2d_readvariableop_resource:N
@dilated_conv2d_stacked8_conv2d_9_biasadd_readvariableop_resource:Z
@dilated_conv2d_stacked8_conv2d_10_conv2d_readvariableop_resource:O
Adilated_conv2d_stacked8_conv2d_10_biasadd_readvariableop_resource:Z
@dilated_conv2d_stacked8_conv2d_11_conv2d_readvariableop_resource:O
Adilated_conv2d_stacked8_conv2d_11_biasadd_readvariableop_resource:Z
@dilated_conv2d_stacked8_conv2d_12_conv2d_readvariableop_resource:O
Adilated_conv2d_stacked8_conv2d_12_biasadd_readvariableop_resource:Z
@dilated_conv2d_stacked8_conv2d_13_conv2d_readvariableop_resource:O
Adilated_conv2d_stacked8_conv2d_13_biasadd_readvariableop_resource:Z
@dilated_conv2d_stacked8_conv2d_14_conv2d_readvariableop_resource:O
Adilated_conv2d_stacked8_conv2d_14_biasadd_readvariableop_resource:Z
@dilated_conv2d_stacked8_conv2d_15_conv2d_readvariableop_resource:O
Adilated_conv2d_stacked8_conv2d_15_biasadd_readvariableop_resource:
identity??8dilated_conv2d_stacked8/conv2d_10/BiasAdd/ReadVariableOp?7dilated_conv2d_stacked8/conv2d_10/Conv2D/ReadVariableOp?8dilated_conv2d_stacked8/conv2d_11/BiasAdd/ReadVariableOp?7dilated_conv2d_stacked8/conv2d_11/Conv2D/ReadVariableOp?8dilated_conv2d_stacked8/conv2d_12/BiasAdd/ReadVariableOp?7dilated_conv2d_stacked8/conv2d_12/Conv2D/ReadVariableOp?8dilated_conv2d_stacked8/conv2d_13/BiasAdd/ReadVariableOp?7dilated_conv2d_stacked8/conv2d_13/Conv2D/ReadVariableOp?8dilated_conv2d_stacked8/conv2d_14/BiasAdd/ReadVariableOp?7dilated_conv2d_stacked8/conv2d_14/Conv2D/ReadVariableOp?8dilated_conv2d_stacked8/conv2d_15/BiasAdd/ReadVariableOp?7dilated_conv2d_stacked8/conv2d_15/Conv2D/ReadVariableOp?7dilated_conv2d_stacked8/conv2d_8/BiasAdd/ReadVariableOp?6dilated_conv2d_stacked8/conv2d_8/Conv2D/ReadVariableOp?7dilated_conv2d_stacked8/conv2d_9/BiasAdd/ReadVariableOp?6dilated_conv2d_stacked8/conv2d_9/Conv2D/ReadVariableOp?
5dilated_conv2d_stacked8/conv2d_8/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      27
5dilated_conv2d_stacked8/conv2d_8/Conv2D/dilation_rate?
Tdilated_conv2d_stacked8/conv2d_8/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2V
Tdilated_conv2d_stacked8/conv2d_8/Conv2D/required_space_to_batch_paddings/input_shape?
Vdilated_conv2d_stacked8/conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2X
Vdilated_conv2d_stacked8/conv2d_8/Conv2D/required_space_to_batch_paddings/base_paddings?
Qdilated_conv2d_stacked8/conv2d_8/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2S
Qdilated_conv2d_stacked8/conv2d_8/Conv2D/required_space_to_batch_paddings/paddings?
Ndilated_conv2d_stacked8/conv2d_8/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2P
Ndilated_conv2d_stacked8/conv2d_8/Conv2D/required_space_to_batch_paddings/crops?
Bdilated_conv2d_stacked8/conv2d_8/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bdilated_conv2d_stacked8/conv2d_8/Conv2D/SpaceToBatchND/block_shape?
?dilated_conv2d_stacked8/conv2d_8/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?dilated_conv2d_stacked8/conv2d_8/Conv2D/SpaceToBatchND/paddings?
6dilated_conv2d_stacked8/conv2d_8/Conv2D/SpaceToBatchNDSpaceToBatchNDinput_1Kdilated_conv2d_stacked8/conv2d_8/Conv2D/SpaceToBatchND/block_shape:output:0Hdilated_conv2d_stacked8/conv2d_8/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:@@28
6dilated_conv2d_stacked8/conv2d_8/Conv2D/SpaceToBatchND?
6dilated_conv2d_stacked8/conv2d_8/Conv2D/ReadVariableOpReadVariableOp?dilated_conv2d_stacked8_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype028
6dilated_conv2d_stacked8/conv2d_8/Conv2D/ReadVariableOp?
'dilated_conv2d_stacked8/conv2d_8/Conv2DConv2D?dilated_conv2d_stacked8/conv2d_8/Conv2D/SpaceToBatchND:output:0>dilated_conv2d_stacked8/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:>>*
paddingVALID*
strides
2)
'dilated_conv2d_stacked8/conv2d_8/Conv2D?
Bdilated_conv2d_stacked8/conv2d_8/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bdilated_conv2d_stacked8/conv2d_8/Conv2D/BatchToSpaceND/block_shape?
<dilated_conv2d_stacked8/conv2d_8/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2>
<dilated_conv2d_stacked8/conv2d_8/Conv2D/BatchToSpaceND/crops?
6dilated_conv2d_stacked8/conv2d_8/Conv2D/BatchToSpaceNDBatchToSpaceND0dilated_conv2d_stacked8/conv2d_8/Conv2D:output:0Kdilated_conv2d_stacked8/conv2d_8/Conv2D/BatchToSpaceND/block_shape:output:0Edilated_conv2d_stacked8/conv2d_8/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:||28
6dilated_conv2d_stacked8/conv2d_8/Conv2D/BatchToSpaceND?
7dilated_conv2d_stacked8/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp@dilated_conv2d_stacked8_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7dilated_conv2d_stacked8/conv2d_8/BiasAdd/ReadVariableOp?
(dilated_conv2d_stacked8/conv2d_8/BiasAddBiasAdd?dilated_conv2d_stacked8/conv2d_8/Conv2D/BatchToSpaceND:output:0?dilated_conv2d_stacked8/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||2*
(dilated_conv2d_stacked8/conv2d_8/BiasAdd?
5dilated_conv2d_stacked8/conv2d_9/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      27
5dilated_conv2d_stacked8/conv2d_9/Conv2D/dilation_rate?
Tdilated_conv2d_stacked8/conv2d_9/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"|   |   2V
Tdilated_conv2d_stacked8/conv2d_9/Conv2D/required_space_to_batch_paddings/input_shape?
Vdilated_conv2d_stacked8/conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2X
Vdilated_conv2d_stacked8/conv2d_9/Conv2D/required_space_to_batch_paddings/base_paddings?
Qdilated_conv2d_stacked8/conv2d_9/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2S
Qdilated_conv2d_stacked8/conv2d_9/Conv2D/required_space_to_batch_paddings/paddings?
Ndilated_conv2d_stacked8/conv2d_9/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2P
Ndilated_conv2d_stacked8/conv2d_9/Conv2D/required_space_to_batch_paddings/crops?
Bdilated_conv2d_stacked8/conv2d_9/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bdilated_conv2d_stacked8/conv2d_9/Conv2D/SpaceToBatchND/block_shape?
?dilated_conv2d_stacked8/conv2d_9/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2A
?dilated_conv2d_stacked8/conv2d_9/Conv2D/SpaceToBatchND/paddings?
6dilated_conv2d_stacked8/conv2d_9/Conv2D/SpaceToBatchNDSpaceToBatchND1dilated_conv2d_stacked8/conv2d_8/BiasAdd:output:0Kdilated_conv2d_stacked8/conv2d_9/Conv2D/SpaceToBatchND/block_shape:output:0Hdilated_conv2d_stacked8/conv2d_9/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:>>28
6dilated_conv2d_stacked8/conv2d_9/Conv2D/SpaceToBatchND?
6dilated_conv2d_stacked8/conv2d_9/Conv2D/ReadVariableOpReadVariableOp?dilated_conv2d_stacked8_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype028
6dilated_conv2d_stacked8/conv2d_9/Conv2D/ReadVariableOp?
'dilated_conv2d_stacked8/conv2d_9/Conv2DConv2D?dilated_conv2d_stacked8/conv2d_9/Conv2D/SpaceToBatchND:output:0>dilated_conv2d_stacked8/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:<<*
paddingVALID*
strides
2)
'dilated_conv2d_stacked8/conv2d_9/Conv2D?
Bdilated_conv2d_stacked8/conv2d_9/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2D
Bdilated_conv2d_stacked8/conv2d_9/Conv2D/BatchToSpaceND/block_shape?
<dilated_conv2d_stacked8/conv2d_9/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2>
<dilated_conv2d_stacked8/conv2d_9/Conv2D/BatchToSpaceND/crops?
6dilated_conv2d_stacked8/conv2d_9/Conv2D/BatchToSpaceNDBatchToSpaceND0dilated_conv2d_stacked8/conv2d_9/Conv2D:output:0Kdilated_conv2d_stacked8/conv2d_9/Conv2D/BatchToSpaceND/block_shape:output:0Edilated_conv2d_stacked8/conv2d_9/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:xx28
6dilated_conv2d_stacked8/conv2d_9/Conv2D/BatchToSpaceND?
7dilated_conv2d_stacked8/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp@dilated_conv2d_stacked8_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7dilated_conv2d_stacked8/conv2d_9/BiasAdd/ReadVariableOp?
(dilated_conv2d_stacked8/conv2d_9/BiasAddBiasAdd?dilated_conv2d_stacked8/conv2d_9/Conv2D/BatchToSpaceND:output:0?dilated_conv2d_stacked8/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx2*
(dilated_conv2d_stacked8/conv2d_9/BiasAdd?
6dilated_conv2d_stacked8/conv2d_10/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      28
6dilated_conv2d_stacked8/conv2d_10/Conv2D/dilation_rate?
Udilated_conv2d_stacked8/conv2d_10/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   2W
Udilated_conv2d_stacked8/conv2d_10/Conv2D/required_space_to_batch_paddings/input_shape?
Wdilated_conv2d_stacked8/conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2Y
Wdilated_conv2d_stacked8/conv2d_10/Conv2D/required_space_to_batch_paddings/base_paddings?
Rdilated_conv2d_stacked8/conv2d_10/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2T
Rdilated_conv2d_stacked8/conv2d_10/Conv2D/required_space_to_batch_paddings/paddings?
Odilated_conv2d_stacked8/conv2d_10/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2Q
Odilated_conv2d_stacked8/conv2d_10/Conv2D/required_space_to_batch_paddings/crops?
Cdilated_conv2d_stacked8/conv2d_10/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_10/Conv2D/SpaceToBatchND/block_shape?
@dilated_conv2d_stacked8/conv2d_10/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2B
@dilated_conv2d_stacked8/conv2d_10/Conv2D/SpaceToBatchND/paddings?
7dilated_conv2d_stacked8/conv2d_10/Conv2D/SpaceToBatchNDSpaceToBatchND1dilated_conv2d_stacked8/conv2d_9/BiasAdd:output:0Ldilated_conv2d_stacked8/conv2d_10/Conv2D/SpaceToBatchND/block_shape:output:0Idilated_conv2d_stacked8/conv2d_10/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<29
7dilated_conv2d_stacked8/conv2d_10/Conv2D/SpaceToBatchND?
7dilated_conv2d_stacked8/conv2d_10/Conv2D/ReadVariableOpReadVariableOp@dilated_conv2d_stacked8_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7dilated_conv2d_stacked8/conv2d_10/Conv2D/ReadVariableOp?
(dilated_conv2d_stacked8/conv2d_10/Conv2DConv2D@dilated_conv2d_stacked8/conv2d_10/Conv2D/SpaceToBatchND:output:0?dilated_conv2d_stacked8/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2*
(dilated_conv2d_stacked8/conv2d_10/Conv2D?
Cdilated_conv2d_stacked8/conv2d_10/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_10/Conv2D/BatchToSpaceND/block_shape?
=dilated_conv2d_stacked8/conv2d_10/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2?
=dilated_conv2d_stacked8/conv2d_10/Conv2D/BatchToSpaceND/crops?
7dilated_conv2d_stacked8/conv2d_10/Conv2D/BatchToSpaceNDBatchToSpaceND1dilated_conv2d_stacked8/conv2d_10/Conv2D:output:0Ldilated_conv2d_stacked8/conv2d_10/Conv2D/BatchToSpaceND/block_shape:output:0Fdilated_conv2d_stacked8/conv2d_10/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt29
7dilated_conv2d_stacked8/conv2d_10/Conv2D/BatchToSpaceND?
8dilated_conv2d_stacked8/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpAdilated_conv2d_stacked8_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8dilated_conv2d_stacked8/conv2d_10/BiasAdd/ReadVariableOp?
)dilated_conv2d_stacked8/conv2d_10/BiasAddBiasAdd@dilated_conv2d_stacked8/conv2d_10/Conv2D/BatchToSpaceND:output:0@dilated_conv2d_stacked8/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2+
)dilated_conv2d_stacked8/conv2d_10/BiasAdd?
6dilated_conv2d_stacked8/conv2d_11/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      28
6dilated_conv2d_stacked8/conv2d_11/Conv2D/dilation_rate?
Udilated_conv2d_stacked8/conv2d_11/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"t   t   2W
Udilated_conv2d_stacked8/conv2d_11/Conv2D/required_space_to_batch_paddings/input_shape?
Wdilated_conv2d_stacked8/conv2d_11/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2Y
Wdilated_conv2d_stacked8/conv2d_11/Conv2D/required_space_to_batch_paddings/base_paddings?
Rdilated_conv2d_stacked8/conv2d_11/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2T
Rdilated_conv2d_stacked8/conv2d_11/Conv2D/required_space_to_batch_paddings/paddings?
Odilated_conv2d_stacked8/conv2d_11/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2Q
Odilated_conv2d_stacked8/conv2d_11/Conv2D/required_space_to_batch_paddings/crops?
Cdilated_conv2d_stacked8/conv2d_11/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_11/Conv2D/SpaceToBatchND/block_shape?
@dilated_conv2d_stacked8/conv2d_11/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2B
@dilated_conv2d_stacked8/conv2d_11/Conv2D/SpaceToBatchND/paddings?
7dilated_conv2d_stacked8/conv2d_11/Conv2D/SpaceToBatchNDSpaceToBatchND2dilated_conv2d_stacked8/conv2d_10/BiasAdd:output:0Ldilated_conv2d_stacked8/conv2d_11/Conv2D/SpaceToBatchND/block_shape:output:0Idilated_conv2d_stacked8/conv2d_11/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:::29
7dilated_conv2d_stacked8/conv2d_11/Conv2D/SpaceToBatchND?
7dilated_conv2d_stacked8/conv2d_11/Conv2D/ReadVariableOpReadVariableOp@dilated_conv2d_stacked8_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7dilated_conv2d_stacked8/conv2d_11/Conv2D/ReadVariableOp?
(dilated_conv2d_stacked8/conv2d_11/Conv2DConv2D@dilated_conv2d_stacked8/conv2d_11/Conv2D/SpaceToBatchND:output:0?dilated_conv2d_stacked8/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:88*
paddingVALID*
strides
2*
(dilated_conv2d_stacked8/conv2d_11/Conv2D?
Cdilated_conv2d_stacked8/conv2d_11/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_11/Conv2D/BatchToSpaceND/block_shape?
=dilated_conv2d_stacked8/conv2d_11/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2?
=dilated_conv2d_stacked8/conv2d_11/Conv2D/BatchToSpaceND/crops?
7dilated_conv2d_stacked8/conv2d_11/Conv2D/BatchToSpaceNDBatchToSpaceND1dilated_conv2d_stacked8/conv2d_11/Conv2D:output:0Ldilated_conv2d_stacked8/conv2d_11/Conv2D/BatchToSpaceND/block_shape:output:0Fdilated_conv2d_stacked8/conv2d_11/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:pp29
7dilated_conv2d_stacked8/conv2d_11/Conv2D/BatchToSpaceND?
8dilated_conv2d_stacked8/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpAdilated_conv2d_stacked8_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8dilated_conv2d_stacked8/conv2d_11/BiasAdd/ReadVariableOp?
)dilated_conv2d_stacked8/conv2d_11/BiasAddBiasAdd@dilated_conv2d_stacked8/conv2d_11/Conv2D/BatchToSpaceND:output:0@dilated_conv2d_stacked8/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp2+
)dilated_conv2d_stacked8/conv2d_11/BiasAdd?
6dilated_conv2d_stacked8/conv2d_12/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      28
6dilated_conv2d_stacked8/conv2d_12/Conv2D/dilation_rate?
Udilated_conv2d_stacked8/conv2d_12/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"p   p   2W
Udilated_conv2d_stacked8/conv2d_12/Conv2D/required_space_to_batch_paddings/input_shape?
Wdilated_conv2d_stacked8/conv2d_12/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2Y
Wdilated_conv2d_stacked8/conv2d_12/Conv2D/required_space_to_batch_paddings/base_paddings?
Rdilated_conv2d_stacked8/conv2d_12/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2T
Rdilated_conv2d_stacked8/conv2d_12/Conv2D/required_space_to_batch_paddings/paddings?
Odilated_conv2d_stacked8/conv2d_12/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2Q
Odilated_conv2d_stacked8/conv2d_12/Conv2D/required_space_to_batch_paddings/crops?
Cdilated_conv2d_stacked8/conv2d_12/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_12/Conv2D/SpaceToBatchND/block_shape?
@dilated_conv2d_stacked8/conv2d_12/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2B
@dilated_conv2d_stacked8/conv2d_12/Conv2D/SpaceToBatchND/paddings?
7dilated_conv2d_stacked8/conv2d_12/Conv2D/SpaceToBatchNDSpaceToBatchND2dilated_conv2d_stacked8/conv2d_11/BiasAdd:output:0Ldilated_conv2d_stacked8/conv2d_12/Conv2D/SpaceToBatchND/block_shape:output:0Idilated_conv2d_stacked8/conv2d_12/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:8829
7dilated_conv2d_stacked8/conv2d_12/Conv2D/SpaceToBatchND?
7dilated_conv2d_stacked8/conv2d_12/Conv2D/ReadVariableOpReadVariableOp@dilated_conv2d_stacked8_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7dilated_conv2d_stacked8/conv2d_12/Conv2D/ReadVariableOp?
(dilated_conv2d_stacked8/conv2d_12/Conv2DConv2D@dilated_conv2d_stacked8/conv2d_12/Conv2D/SpaceToBatchND:output:0?dilated_conv2d_stacked8/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:66*
paddingVALID*
strides
2*
(dilated_conv2d_stacked8/conv2d_12/Conv2D?
Cdilated_conv2d_stacked8/conv2d_12/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_12/Conv2D/BatchToSpaceND/block_shape?
=dilated_conv2d_stacked8/conv2d_12/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2?
=dilated_conv2d_stacked8/conv2d_12/Conv2D/BatchToSpaceND/crops?
7dilated_conv2d_stacked8/conv2d_12/Conv2D/BatchToSpaceNDBatchToSpaceND1dilated_conv2d_stacked8/conv2d_12/Conv2D:output:0Ldilated_conv2d_stacked8/conv2d_12/Conv2D/BatchToSpaceND/block_shape:output:0Fdilated_conv2d_stacked8/conv2d_12/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:ll29
7dilated_conv2d_stacked8/conv2d_12/Conv2D/BatchToSpaceND?
8dilated_conv2d_stacked8/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpAdilated_conv2d_stacked8_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8dilated_conv2d_stacked8/conv2d_12/BiasAdd/ReadVariableOp?
)dilated_conv2d_stacked8/conv2d_12/BiasAddBiasAdd@dilated_conv2d_stacked8/conv2d_12/Conv2D/BatchToSpaceND:output:0@dilated_conv2d_stacked8/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:ll2+
)dilated_conv2d_stacked8/conv2d_12/BiasAdd?
6dilated_conv2d_stacked8/conv2d_13/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      28
6dilated_conv2d_stacked8/conv2d_13/Conv2D/dilation_rate?
Udilated_conv2d_stacked8/conv2d_13/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"l   l   2W
Udilated_conv2d_stacked8/conv2d_13/Conv2D/required_space_to_batch_paddings/input_shape?
Wdilated_conv2d_stacked8/conv2d_13/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2Y
Wdilated_conv2d_stacked8/conv2d_13/Conv2D/required_space_to_batch_paddings/base_paddings?
Rdilated_conv2d_stacked8/conv2d_13/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2T
Rdilated_conv2d_stacked8/conv2d_13/Conv2D/required_space_to_batch_paddings/paddings?
Odilated_conv2d_stacked8/conv2d_13/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2Q
Odilated_conv2d_stacked8/conv2d_13/Conv2D/required_space_to_batch_paddings/crops?
Cdilated_conv2d_stacked8/conv2d_13/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_13/Conv2D/SpaceToBatchND/block_shape?
@dilated_conv2d_stacked8/conv2d_13/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2B
@dilated_conv2d_stacked8/conv2d_13/Conv2D/SpaceToBatchND/paddings?
7dilated_conv2d_stacked8/conv2d_13/Conv2D/SpaceToBatchNDSpaceToBatchND2dilated_conv2d_stacked8/conv2d_12/BiasAdd:output:0Ldilated_conv2d_stacked8/conv2d_13/Conv2D/SpaceToBatchND/block_shape:output:0Idilated_conv2d_stacked8/conv2d_13/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:6629
7dilated_conv2d_stacked8/conv2d_13/Conv2D/SpaceToBatchND?
7dilated_conv2d_stacked8/conv2d_13/Conv2D/ReadVariableOpReadVariableOp@dilated_conv2d_stacked8_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7dilated_conv2d_stacked8/conv2d_13/Conv2D/ReadVariableOp?
(dilated_conv2d_stacked8/conv2d_13/Conv2DConv2D@dilated_conv2d_stacked8/conv2d_13/Conv2D/SpaceToBatchND:output:0?dilated_conv2d_stacked8/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:44*
paddingVALID*
strides
2*
(dilated_conv2d_stacked8/conv2d_13/Conv2D?
Cdilated_conv2d_stacked8/conv2d_13/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_13/Conv2D/BatchToSpaceND/block_shape?
=dilated_conv2d_stacked8/conv2d_13/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2?
=dilated_conv2d_stacked8/conv2d_13/Conv2D/BatchToSpaceND/crops?
7dilated_conv2d_stacked8/conv2d_13/Conv2D/BatchToSpaceNDBatchToSpaceND1dilated_conv2d_stacked8/conv2d_13/Conv2D:output:0Ldilated_conv2d_stacked8/conv2d_13/Conv2D/BatchToSpaceND/block_shape:output:0Fdilated_conv2d_stacked8/conv2d_13/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:hh29
7dilated_conv2d_stacked8/conv2d_13/Conv2D/BatchToSpaceND?
8dilated_conv2d_stacked8/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpAdilated_conv2d_stacked8_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8dilated_conv2d_stacked8/conv2d_13/BiasAdd/ReadVariableOp?
)dilated_conv2d_stacked8/conv2d_13/BiasAddBiasAdd@dilated_conv2d_stacked8/conv2d_13/Conv2D/BatchToSpaceND:output:0@dilated_conv2d_stacked8/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:hh2+
)dilated_conv2d_stacked8/conv2d_13/BiasAdd?
6dilated_conv2d_stacked8/conv2d_14/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      28
6dilated_conv2d_stacked8/conv2d_14/Conv2D/dilation_rate?
Udilated_conv2d_stacked8/conv2d_14/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"h   h   2W
Udilated_conv2d_stacked8/conv2d_14/Conv2D/required_space_to_batch_paddings/input_shape?
Wdilated_conv2d_stacked8/conv2d_14/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2Y
Wdilated_conv2d_stacked8/conv2d_14/Conv2D/required_space_to_batch_paddings/base_paddings?
Rdilated_conv2d_stacked8/conv2d_14/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2T
Rdilated_conv2d_stacked8/conv2d_14/Conv2D/required_space_to_batch_paddings/paddings?
Odilated_conv2d_stacked8/conv2d_14/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2Q
Odilated_conv2d_stacked8/conv2d_14/Conv2D/required_space_to_batch_paddings/crops?
Cdilated_conv2d_stacked8/conv2d_14/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_14/Conv2D/SpaceToBatchND/block_shape?
@dilated_conv2d_stacked8/conv2d_14/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2B
@dilated_conv2d_stacked8/conv2d_14/Conv2D/SpaceToBatchND/paddings?
7dilated_conv2d_stacked8/conv2d_14/Conv2D/SpaceToBatchNDSpaceToBatchND2dilated_conv2d_stacked8/conv2d_13/BiasAdd:output:0Ldilated_conv2d_stacked8/conv2d_14/Conv2D/SpaceToBatchND/block_shape:output:0Idilated_conv2d_stacked8/conv2d_14/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:4429
7dilated_conv2d_stacked8/conv2d_14/Conv2D/SpaceToBatchND?
7dilated_conv2d_stacked8/conv2d_14/Conv2D/ReadVariableOpReadVariableOp@dilated_conv2d_stacked8_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7dilated_conv2d_stacked8/conv2d_14/Conv2D/ReadVariableOp?
(dilated_conv2d_stacked8/conv2d_14/Conv2DConv2D@dilated_conv2d_stacked8/conv2d_14/Conv2D/SpaceToBatchND:output:0?dilated_conv2d_stacked8/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:22*
paddingVALID*
strides
2*
(dilated_conv2d_stacked8/conv2d_14/Conv2D?
Cdilated_conv2d_stacked8/conv2d_14/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_14/Conv2D/BatchToSpaceND/block_shape?
=dilated_conv2d_stacked8/conv2d_14/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2?
=dilated_conv2d_stacked8/conv2d_14/Conv2D/BatchToSpaceND/crops?
7dilated_conv2d_stacked8/conv2d_14/Conv2D/BatchToSpaceNDBatchToSpaceND1dilated_conv2d_stacked8/conv2d_14/Conv2D:output:0Ldilated_conv2d_stacked8/conv2d_14/Conv2D/BatchToSpaceND/block_shape:output:0Fdilated_conv2d_stacked8/conv2d_14/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:dd29
7dilated_conv2d_stacked8/conv2d_14/Conv2D/BatchToSpaceND?
8dilated_conv2d_stacked8/conv2d_14/BiasAdd/ReadVariableOpReadVariableOpAdilated_conv2d_stacked8_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8dilated_conv2d_stacked8/conv2d_14/BiasAdd/ReadVariableOp?
)dilated_conv2d_stacked8/conv2d_14/BiasAddBiasAdd@dilated_conv2d_stacked8/conv2d_14/Conv2D/BatchToSpaceND:output:0@dilated_conv2d_stacked8/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:dd2+
)dilated_conv2d_stacked8/conv2d_14/BiasAdd?
6dilated_conv2d_stacked8/conv2d_15/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      28
6dilated_conv2d_stacked8/conv2d_15/Conv2D/dilation_rate?
Udilated_conv2d_stacked8/conv2d_15/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   2W
Udilated_conv2d_stacked8/conv2d_15/Conv2D/required_space_to_batch_paddings/input_shape?
Wdilated_conv2d_stacked8/conv2d_15/Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2Y
Wdilated_conv2d_stacked8/conv2d_15/Conv2D/required_space_to_batch_paddings/base_paddings?
Rdilated_conv2d_stacked8/conv2d_15/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2T
Rdilated_conv2d_stacked8/conv2d_15/Conv2D/required_space_to_batch_paddings/paddings?
Odilated_conv2d_stacked8/conv2d_15/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2Q
Odilated_conv2d_stacked8/conv2d_15/Conv2D/required_space_to_batch_paddings/crops?
Cdilated_conv2d_stacked8/conv2d_15/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_15/Conv2D/SpaceToBatchND/block_shape?
@dilated_conv2d_stacked8/conv2d_15/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2B
@dilated_conv2d_stacked8/conv2d_15/Conv2D/SpaceToBatchND/paddings?
7dilated_conv2d_stacked8/conv2d_15/Conv2D/SpaceToBatchNDSpaceToBatchND2dilated_conv2d_stacked8/conv2d_14/BiasAdd:output:0Ldilated_conv2d_stacked8/conv2d_15/Conv2D/SpaceToBatchND/block_shape:output:0Idilated_conv2d_stacked8/conv2d_15/Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:2229
7dilated_conv2d_stacked8/conv2d_15/Conv2D/SpaceToBatchND?
7dilated_conv2d_stacked8/conv2d_15/Conv2D/ReadVariableOpReadVariableOp@dilated_conv2d_stacked8_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7dilated_conv2d_stacked8/conv2d_15/Conv2D/ReadVariableOp?
(dilated_conv2d_stacked8/conv2d_15/Conv2DConv2D@dilated_conv2d_stacked8/conv2d_15/Conv2D/SpaceToBatchND:output:0?dilated_conv2d_stacked8/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:00*
paddingVALID*
strides
2*
(dilated_conv2d_stacked8/conv2d_15/Conv2D?
Cdilated_conv2d_stacked8/conv2d_15/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cdilated_conv2d_stacked8/conv2d_15/Conv2D/BatchToSpaceND/block_shape?
=dilated_conv2d_stacked8/conv2d_15/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2?
=dilated_conv2d_stacked8/conv2d_15/Conv2D/BatchToSpaceND/crops?
7dilated_conv2d_stacked8/conv2d_15/Conv2D/BatchToSpaceNDBatchToSpaceND1dilated_conv2d_stacked8/conv2d_15/Conv2D:output:0Ldilated_conv2d_stacked8/conv2d_15/Conv2D/BatchToSpaceND/block_shape:output:0Fdilated_conv2d_stacked8/conv2d_15/Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:``29
7dilated_conv2d_stacked8/conv2d_15/Conv2D/BatchToSpaceND?
8dilated_conv2d_stacked8/conv2d_15/BiasAdd/ReadVariableOpReadVariableOpAdilated_conv2d_stacked8_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8dilated_conv2d_stacked8/conv2d_15/BiasAdd/ReadVariableOp?
)dilated_conv2d_stacked8/conv2d_15/BiasAddBiasAdd@dilated_conv2d_stacked8/conv2d_15/Conv2D/BatchToSpaceND:output:0@dilated_conv2d_stacked8/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:``2+
)dilated_conv2d_stacked8/conv2d_15/BiasAdd?
IdentityIdentity2dilated_conv2d_stacked8/conv2d_15/BiasAdd:output:09^dilated_conv2d_stacked8/conv2d_10/BiasAdd/ReadVariableOp8^dilated_conv2d_stacked8/conv2d_10/Conv2D/ReadVariableOp9^dilated_conv2d_stacked8/conv2d_11/BiasAdd/ReadVariableOp8^dilated_conv2d_stacked8/conv2d_11/Conv2D/ReadVariableOp9^dilated_conv2d_stacked8/conv2d_12/BiasAdd/ReadVariableOp8^dilated_conv2d_stacked8/conv2d_12/Conv2D/ReadVariableOp9^dilated_conv2d_stacked8/conv2d_13/BiasAdd/ReadVariableOp8^dilated_conv2d_stacked8/conv2d_13/Conv2D/ReadVariableOp9^dilated_conv2d_stacked8/conv2d_14/BiasAdd/ReadVariableOp8^dilated_conv2d_stacked8/conv2d_14/Conv2D/ReadVariableOp9^dilated_conv2d_stacked8/conv2d_15/BiasAdd/ReadVariableOp8^dilated_conv2d_stacked8/conv2d_15/Conv2D/ReadVariableOp8^dilated_conv2d_stacked8/conv2d_8/BiasAdd/ReadVariableOp7^dilated_conv2d_stacked8/conv2d_8/Conv2D/ReadVariableOp8^dilated_conv2d_stacked8/conv2d_9/BiasAdd/ReadVariableOp7^dilated_conv2d_stacked8/conv2d_9/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 2t
8dilated_conv2d_stacked8/conv2d_10/BiasAdd/ReadVariableOp8dilated_conv2d_stacked8/conv2d_10/BiasAdd/ReadVariableOp2r
7dilated_conv2d_stacked8/conv2d_10/Conv2D/ReadVariableOp7dilated_conv2d_stacked8/conv2d_10/Conv2D/ReadVariableOp2t
8dilated_conv2d_stacked8/conv2d_11/BiasAdd/ReadVariableOp8dilated_conv2d_stacked8/conv2d_11/BiasAdd/ReadVariableOp2r
7dilated_conv2d_stacked8/conv2d_11/Conv2D/ReadVariableOp7dilated_conv2d_stacked8/conv2d_11/Conv2D/ReadVariableOp2t
8dilated_conv2d_stacked8/conv2d_12/BiasAdd/ReadVariableOp8dilated_conv2d_stacked8/conv2d_12/BiasAdd/ReadVariableOp2r
7dilated_conv2d_stacked8/conv2d_12/Conv2D/ReadVariableOp7dilated_conv2d_stacked8/conv2d_12/Conv2D/ReadVariableOp2t
8dilated_conv2d_stacked8/conv2d_13/BiasAdd/ReadVariableOp8dilated_conv2d_stacked8/conv2d_13/BiasAdd/ReadVariableOp2r
7dilated_conv2d_stacked8/conv2d_13/Conv2D/ReadVariableOp7dilated_conv2d_stacked8/conv2d_13/Conv2D/ReadVariableOp2t
8dilated_conv2d_stacked8/conv2d_14/BiasAdd/ReadVariableOp8dilated_conv2d_stacked8/conv2d_14/BiasAdd/ReadVariableOp2r
7dilated_conv2d_stacked8/conv2d_14/Conv2D/ReadVariableOp7dilated_conv2d_stacked8/conv2d_14/Conv2D/ReadVariableOp2t
8dilated_conv2d_stacked8/conv2d_15/BiasAdd/ReadVariableOp8dilated_conv2d_stacked8/conv2d_15/BiasAdd/ReadVariableOp2r
7dilated_conv2d_stacked8/conv2d_15/Conv2D/ReadVariableOp7dilated_conv2d_stacked8/conv2d_15/Conv2D/ReadVariableOp2r
7dilated_conv2d_stacked8/conv2d_8/BiasAdd/ReadVariableOp7dilated_conv2d_stacked8/conv2d_8/BiasAdd/ReadVariableOp2p
6dilated_conv2d_stacked8/conv2d_8/Conv2D/ReadVariableOp6dilated_conv2d_stacked8/conv2d_8/Conv2D/ReadVariableOp2r
7dilated_conv2d_stacked8/conv2d_9/BiasAdd/ReadVariableOp7dilated_conv2d_stacked8/conv2d_9/BiasAdd/ReadVariableOp2p
6dilated_conv2d_stacked8/conv2d_9/Conv2D/ReadVariableOp6dilated_conv2d_stacked8/conv2d_9/Conv2D/ReadVariableOp:Q M
(
_output_shapes
:??
!
_user_specified_name	input_1
?
?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_36447

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:xx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:xx
 
_user_specified_nameinputs
?
?
7__inference_dilated_conv2d_stacked8_layer_call_fn_37303

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_365892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs
?
?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_37421

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
Conv2D/dilation_rate?
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"x   x   25
3Conv2D/required_space_to_batch_paddings/input_shape?
5Conv2D/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*)
value B"                27
5Conv2D/required_space_to_batch_paddings/base_paddings?
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                22
0Conv2D/required_space_to_batch_paddings/paddings?
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2/
-Conv2D/required_space_to_batch_paddings/crops?
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/SpaceToBatchND/block_shape?
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"                2 
Conv2D/SpaceToBatchND/paddings?
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*&
_output_shapes
:<<2
Conv2D/SpaceToBatchND?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:::*
paddingVALID*
strides
2
Conv2D?
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!Conv2D/BatchToSpaceND/block_shape?
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                2
Conv2D/BatchToSpaceND/crops?
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*&
_output_shapes
:tt2
Conv2D/BatchToSpaceND?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D/BatchToSpaceND:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:xx: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:xx
 
_user_specified_nameinputs
?
?
7__inference_dilated_conv2d_stacked8_layer_call_fn_37340

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:``*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_367872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??<
	conv2d_15/
StatefulPartitionedCall:0``tensorflow/serving/predict:??
?q
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

trainable_variables
regularization_losses
	variables
	keras_api

signatures
*l&call_and_return_all_conditional_losses
m__call__
n_default_save_signature"?m
_tf_keras_network?m{"name": "dilated_conv2d_stacked8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "dilated_conv2d_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_15", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "dilated_conv2d_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_12", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_13", "inbound_nodes": [[["conv2d_12", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_14", "inbound_nodes": [[["conv2d_13", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["conv2d_14", 0, 0, {}]]], "shared_object_id": 24}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_15", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*o&call_and_return_all_conditional_losses
p__call__"?	
_tf_keras_layer?	{"name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*q&call_and_return_all_conditional_losses
r__call__"?	
_tf_keras_layer?	{"name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 124, 124, 12]}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
*s&call_and_return_all_conditional_losses
t__call__"?	
_tf_keras_layer?	{"name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_9", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 120, 120, 12]}}
?

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
*u&call_and_return_all_conditional_losses
v__call__"?	
_tf_keras_layer?	{"name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_10", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 116, 116, 12]}}
?

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
*w&call_and_return_all_conditional_losses
x__call__"?	
_tf_keras_layer?	{"name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_11", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 112, 112, 12]}}
?

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
*y&call_and_return_all_conditional_losses
z__call__"?	
_tf_keras_layer?	{"name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_12", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 108, 108, 12]}}
?

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
*{&call_and_return_all_conditional_losses
|__call__"?	
_tf_keras_layer?	{"name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_13", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 104, 104, 12]}}
?

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
*}&call_and_return_all_conditional_losses
~__call__"?	
_tf_keras_layer?	{"name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2, 2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_14", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100, 100, 12]}}
?
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
413
914
:15"
trackable_list_wrapper
?

?layers
@layer_metrics

trainable_variables
Ametrics
Blayer_regularization_losses
regularization_losses
	variables
Cnon_trainable_variables
m__call__
n_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
):'2conv2d_8/kernel
:2conv2d_8/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Dlayers
Elayer_metrics
trainable_variables
Fmetrics
Glayer_regularization_losses
regularization_losses
	variables
Hnon_trainable_variables
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_9/kernel
:2conv2d_9/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Ilayers
Jlayer_metrics
trainable_variables
Kmetrics
Llayer_regularization_losses
regularization_losses
	variables
Mnon_trainable_variables
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_10/kernel
:2conv2d_10/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Nlayers
Olayer_metrics
trainable_variables
Pmetrics
Qlayer_regularization_losses
regularization_losses
	variables
Rnon_trainable_variables
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_11/kernel
:2conv2d_11/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?

Slayers
Tlayer_metrics
#trainable_variables
Umetrics
Vlayer_regularization_losses
$regularization_losses
%	variables
Wnon_trainable_variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_12/kernel
:2conv2d_12/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?

Xlayers
Ylayer_metrics
)trainable_variables
Zmetrics
[layer_regularization_losses
*regularization_losses
+	variables
\non_trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_13/kernel
:2conv2d_13/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?

]layers
^layer_metrics
/trainable_variables
_metrics
`layer_regularization_losses
0regularization_losses
1	variables
anon_trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_14/kernel
:2conv2d_14/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?

blayers
clayer_metrics
5trainable_variables
dmetrics
elayer_regularization_losses
6regularization_losses
7	variables
fnon_trainable_variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_15/kernel
:2conv2d_15/bias
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?

glayers
hlayer_metrics
;trainable_variables
imetrics
jlayer_regularization_losses
<regularization_losses
=	variables
knon_trainable_variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
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
?2?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_37126
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_37266
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_36903
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_36947?
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
7__inference_dilated_conv2d_stacked8_layer_call_fn_36624
7__inference_dilated_conv2d_stacked8_layer_call_fn_37303
7__inference_dilated_conv2d_stacked8_layer_call_fn_37340
7__inference_dilated_conv2d_stacked8_layer_call_fn_36859?
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
 __inference__wrapped_model_36365?
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
annotations? *'?$
"?
input_1??
?2?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_37361?
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
(__inference_conv2d_8_layer_call_fn_37370?
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_37391?
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
(__inference_conv2d_9_layer_call_fn_37400?
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
D__inference_conv2d_10_layer_call_and_return_conditional_losses_37421?
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
)__inference_conv2d_10_layer_call_fn_37430?
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
D__inference_conv2d_11_layer_call_and_return_conditional_losses_37451?
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
)__inference_conv2d_11_layer_call_fn_37460?
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
D__inference_conv2d_12_layer_call_and_return_conditional_losses_37481?
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
)__inference_conv2d_12_layer_call_fn_37490?
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
D__inference_conv2d_13_layer_call_and_return_conditional_losses_37511?
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
)__inference_conv2d_13_layer_call_fn_37520?
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
D__inference_conv2d_14_layer_call_and_return_conditional_losses_37541?
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
)__inference_conv2d_14_layer_call_fn_37550?
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
D__inference_conv2d_15_layer_call_and_return_conditional_losses_37571?
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
)__inference_conv2d_15_layer_call_fn_37580?
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
#__inference_signature_wrapper_36986input_1"?
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
 __inference__wrapped_model_36365{!"'(-.349:1?.
'?$
"?
input_1??
? "4?1
/
	conv2d_15"?
	conv2d_15``?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_37421Z.?+
$?!
?
inputsxx
? "$?!
?
0tt
? z
)__inference_conv2d_10_layer_call_fn_37430M.?+
$?!
?
inputsxx
? "?tt?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_37451Z!".?+
$?!
?
inputstt
? "$?!
?
0pp
? z
)__inference_conv2d_11_layer_call_fn_37460M!".?+
$?!
?
inputstt
? "?pp?
D__inference_conv2d_12_layer_call_and_return_conditional_losses_37481Z'(.?+
$?!
?
inputspp
? "$?!
?
0ll
? z
)__inference_conv2d_12_layer_call_fn_37490M'(.?+
$?!
?
inputspp
? "?ll?
D__inference_conv2d_13_layer_call_and_return_conditional_losses_37511Z-..?+
$?!
?
inputsll
? "$?!
?
0hh
? z
)__inference_conv2d_13_layer_call_fn_37520M-..?+
$?!
?
inputsll
? "?hh?
D__inference_conv2d_14_layer_call_and_return_conditional_losses_37541Z34.?+
$?!
?
inputshh
? "$?!
?
0dd
? z
)__inference_conv2d_14_layer_call_fn_37550M34.?+
$?!
?
inputshh
? "?dd?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_37571Z9:.?+
$?!
?
inputsdd
? "$?!
?
0``
? z
)__inference_conv2d_15_layer_call_fn_37580M9:.?+
$?!
?
inputsdd
? "?``?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_37361\0?-
&?#
!?
inputs??
? "$?!
?
0||
? {
(__inference_conv2d_8_layer_call_fn_37370O0?-
&?#
!?
inputs??
? "?||?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_37391Z.?+
$?!
?
inputs||
? "$?!
?
0xx
? y
(__inference_conv2d_9_layer_call_fn_37400M.?+
$?!
?
inputs||
? "?xx?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_36903s!"'(-.349:9?6
/?,
"?
input_1??
p 

 
? "$?!
?
0``
? ?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_36947s!"'(-.349:9?6
/?,
"?
input_1??
p

 
? "$?!
?
0``
? ?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_37126r!"'(-.349:8?5
.?+
!?
inputs??
p 

 
? "$?!
?
0``
? ?
R__inference_dilated_conv2d_stacked8_layer_call_and_return_conditional_losses_37266r!"'(-.349:8?5
.?+
!?
inputs??
p

 
? "$?!
?
0``
? ?
7__inference_dilated_conv2d_stacked8_layer_call_fn_36624f!"'(-.349:9?6
/?,
"?
input_1??
p 

 
? "?``?
7__inference_dilated_conv2d_stacked8_layer_call_fn_36859f!"'(-.349:9?6
/?,
"?
input_1??
p

 
? "?``?
7__inference_dilated_conv2d_stacked8_layer_call_fn_37303e!"'(-.349:8?5
.?+
!?
inputs??
p 

 
? "?``?
7__inference_dilated_conv2d_stacked8_layer_call_fn_37340e!"'(-.349:8?5
.?+
!?
inputs??
p

 
? "?``?
#__inference_signature_wrapper_36986?!"'(-.349:<?9
? 
2?/
-
input_1"?
input_1??"4?1
/
	conv2d_15"?
	conv2d_15``