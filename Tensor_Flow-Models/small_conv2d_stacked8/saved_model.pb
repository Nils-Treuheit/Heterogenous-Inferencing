¸î	
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Ó

conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:	*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:	*
dtype0

conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
:	*
dtype0

conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_29/kernel
}
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:	*
dtype0

conv2d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_30/kernel
}
$conv2d_30/kernel/Read/ReadVariableOpReadVariableOpconv2d_30/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_30/bias
m
"conv2d_30/bias/Read/ReadVariableOpReadVariableOpconv2d_30/bias*
_output_shapes
:	*
dtype0

conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_31/kernel
}
$conv2d_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_31/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_31/bias
m
"conv2d_31/bias/Read/ReadVariableOpReadVariableOpconv2d_31/bias*
_output_shapes
:	*
dtype0

conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_32/kernel
}
$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_32/bias
m
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes
:	*
dtype0

conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:	*
dtype0

conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:		*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:	*
dtype0

NoOpNoOp
â'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*'
value'B' B'
§
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
­

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
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Dlayers
Elayer_metrics
trainable_variables
Fmetrics
Glayer_regularization_losses
regularization_losses
	variables
Hnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_28/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_28/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Ilayers
Jlayer_metrics
trainable_variables
Kmetrics
Llayer_regularization_losses
regularization_losses
	variables
Mnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_29/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_29/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Nlayers
Olayer_metrics
trainable_variables
Pmetrics
Qlayer_regularization_losses
regularization_losses
	variables
Rnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_30/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_30/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
­

Slayers
Tlayer_metrics
#trainable_variables
Umetrics
Vlayer_regularization_losses
$regularization_losses
%	variables
Wnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_31/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_31/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
­

Xlayers
Ylayer_metrics
)trainable_variables
Zmetrics
[layer_regularization_losses
*regularization_losses
+	variables
\non_trainable_variables
\Z
VARIABLE_VALUEconv2d_32/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_32/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
­

]layers
^layer_metrics
/trainable_variables
_metrics
`layer_regularization_losses
0regularization_losses
1	variables
anon_trainable_variables
\Z
VARIABLE_VALUEconv2d_33/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_33/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
­

blayers
clayer_metrics
5trainable_variables
dmetrics
elayer_regularization_losses
6regularization_losses
7	variables
fnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_34/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_34/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
­

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
:*
dtype0*
shape:
à
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_53402
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ú
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp$conv2d_30/kernel/Read/ReadVariableOp"conv2d_30/bias/Read/ReadVariableOp$conv2d_31/kernel/Read/ReadVariableOp"conv2d_31/bias/Read/ReadVariableOp$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOpConst*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_53803
µ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/bias*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_53861¿
É
á
5__inference_small_conv2d_stacked8_layer_call_fn_53580

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCallº
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
:pp	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_532032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs


)__inference_conv2d_31_layer_call_fn_53675

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:vv	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_529502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:vv	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:xx	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:xx	
 
_user_specified_nameinputs
¢J
ÿ
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53454

inputsB
(conv2d_27_conv2d_readvariableop_resource:	7
)conv2d_27_biasadd_readvariableop_resource:	B
(conv2d_28_conv2d_readvariableop_resource:		7
)conv2d_28_biasadd_readvariableop_resource:	B
(conv2d_29_conv2d_readvariableop_resource:		7
)conv2d_29_biasadd_readvariableop_resource:	B
(conv2d_30_conv2d_readvariableop_resource:		7
)conv2d_30_biasadd_readvariableop_resource:	B
(conv2d_31_conv2d_readvariableop_resource:		7
)conv2d_31_biasadd_readvariableop_resource:	B
(conv2d_32_conv2d_readvariableop_resource:		7
)conv2d_32_biasadd_readvariableop_resource:	B
(conv2d_33_conv2d_readvariableop_resource:		7
)conv2d_33_biasadd_readvariableop_resource:	B
(conv2d_34_conv2d_readvariableop_resource:		7
)conv2d_34_biasadd_readvariableop_resource:	
identity¢ conv2d_27/BiasAdd/ReadVariableOp¢conv2d_27/Conv2D/ReadVariableOp¢ conv2d_28/BiasAdd/ReadVariableOp¢conv2d_28/Conv2D/ReadVariableOp¢ conv2d_29/BiasAdd/ReadVariableOp¢conv2d_29/Conv2D/ReadVariableOp¢ conv2d_30/BiasAdd/ReadVariableOp¢conv2d_30/Conv2D/ReadVariableOp¢ conv2d_31/BiasAdd/ReadVariableOp¢conv2d_31/Conv2D/ReadVariableOp¢ conv2d_32/BiasAdd/ReadVariableOp¢conv2d_32/Conv2D/ReadVariableOp¢ conv2d_33/BiasAdd/ReadVariableOp¢conv2d_33/Conv2D/ReadVariableOp¢ conv2d_34/BiasAdd/ReadVariableOp¢conv2d_34/Conv2D/ReadVariableOp³
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02!
conv2d_27/Conv2D/ReadVariableOp¹
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	*
paddingVALID*
strides
2
conv2d_27/Conv2Dª
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp§
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	2
conv2d_27/BiasAdd³
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_28/Conv2D/ReadVariableOpÍ
conv2d_28/Conv2DConv2Dconv2d_27/BiasAdd:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	*
paddingVALID*
strides
2
conv2d_28/Conv2Dª
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp§
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	2
conv2d_28/BiasAdd³
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_29/Conv2D/ReadVariableOpÍ
conv2d_29/Conv2DConv2Dconv2d_28/BiasAdd:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2
conv2d_29/Conv2Dª
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp§
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2
conv2d_29/BiasAdd³
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_30/Conv2D/ReadVariableOpÍ
conv2d_30/Conv2DConv2Dconv2d_29/BiasAdd:output:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	*
paddingVALID*
strides
2
conv2d_30/Conv2Dª
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp§
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	2
conv2d_30/BiasAdd³
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_31/Conv2D/ReadVariableOpÍ
conv2d_31/Conv2DConv2Dconv2d_30/BiasAdd:output:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	*
paddingVALID*
strides
2
conv2d_31/Conv2Dª
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp§
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	2
conv2d_31/BiasAdd³
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_32/Conv2D/ReadVariableOpÍ
conv2d_32/Conv2DConv2Dconv2d_31/BiasAdd:output:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2
conv2d_32/Conv2Dª
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp§
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2
conv2d_32/BiasAdd³
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_33/Conv2D/ReadVariableOpÍ
conv2d_33/Conv2DConv2Dconv2d_32/BiasAdd:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	*
paddingVALID*
strides
2
conv2d_33/Conv2Dª
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp§
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	2
conv2d_33/BiasAdd³
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_34/Conv2D/ReadVariableOpÍ
conv2d_34/Conv2DConv2Dconv2d_33/BiasAdd:output:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	*
paddingVALID*
strides
2
conv2d_34/Conv2Dª
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp§
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	2
conv2d_34/BiasAdd
IdentityIdentityconv2d_34/BiasAdd:output:0!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs


ý
D__inference_conv2d_29_layer_call_and_return_conditional_losses_52918

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:zz	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:||	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:||	
 
_user_specified_nameinputs


)__inference_conv2d_28_layer_call_fn_53618

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_529022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:||	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:~~	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:~~	
 
_user_specified_nameinputs


ý
D__inference_conv2d_31_layer_call_and_return_conditional_losses_53666

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:vv	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:xx	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:xx	
 
_user_specified_nameinputs


ý
D__inference_conv2d_34_layer_call_and_return_conditional_losses_53723

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:rr	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:rr	
 
_user_specified_nameinputs
å,
à
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53363
input_1)
conv2d_27_53322:	
conv2d_27_53324:	)
conv2d_28_53327:		
conv2d_28_53329:	)
conv2d_29_53332:		
conv2d_29_53334:	)
conv2d_30_53337:		
conv2d_30_53339:	)
conv2d_31_53342:		
conv2d_31_53344:	)
conv2d_32_53347:		
conv2d_32_53349:	)
conv2d_33_53352:		
conv2d_33_53354:	)
conv2d_34_53357:		
conv2d_34_53359:	
identity¢!conv2d_27/StatefulPartitionedCall¢!conv2d_28/StatefulPartitionedCall¢!conv2d_29/StatefulPartitionedCall¢!conv2d_30/StatefulPartitionedCall¢!conv2d_31/StatefulPartitionedCall¢!conv2d_32/StatefulPartitionedCall¢!conv2d_33/StatefulPartitionedCall¢!conv2d_34/StatefulPartitionedCall
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_27_53322conv2d_27_53324*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:~~	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_528862#
!conv2d_27/StatefulPartitionedCall¹
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_53327conv2d_28_53329*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_529022#
!conv2d_28/StatefulPartitionedCall¹
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_53332conv2d_29_53334*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_529182#
!conv2d_29/StatefulPartitionedCall¹
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_53337conv2d_30_53339*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_529342#
!conv2d_30/StatefulPartitionedCall¹
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0conv2d_31_53342conv2d_31_53344*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:vv	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_529502#
!conv2d_31/StatefulPartitionedCall¹
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_53347conv2d_32_53349*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_529662#
!conv2d_32/StatefulPartitionedCall¹
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0conv2d_33_53352conv2d_33_53354*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:rr	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_529822#
!conv2d_33/StatefulPartitionedCall¹
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_53357conv2d_34_53359*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_529982#
!conv2d_34/StatefulPartitionedCall
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ý
D__inference_conv2d_34_layer_call_and_return_conditional_losses_52998

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:rr	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:rr	
 
_user_specified_nameinputs
Ì
â
5__inference_small_conv2d_stacked8_layer_call_fn_53275
input_1!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCall»
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
:pp	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_532032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ý
D__inference_conv2d_27_layer_call_and_return_conditional_losses_52886

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:~~	2

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


)__inference_conv2d_33_layer_call_fn_53713

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:rr	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_529822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:rr	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:tt	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:tt	
 
_user_specified_nameinputs
É
á
5__inference_small_conv2d_stacked8_layer_call_fn_53543

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCallº
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
:pp	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_530052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs


)__inference_conv2d_29_layer_call_fn_53637

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_529182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:zz	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:||	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:||	
 
_user_specified_nameinputs


ý
D__inference_conv2d_33_layer_call_and_return_conditional_losses_52982

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:rr	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:tt	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:tt	
 
_user_specified_nameinputs
£

)__inference_conv2d_27_layer_call_fn_53599

inputs!
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:~~	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_528862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:~~	2

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
â,
ß
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53005

inputs)
conv2d_27_52887:	
conv2d_27_52889:	)
conv2d_28_52903:		
conv2d_28_52905:	)
conv2d_29_52919:		
conv2d_29_52921:	)
conv2d_30_52935:		
conv2d_30_52937:	)
conv2d_31_52951:		
conv2d_31_52953:	)
conv2d_32_52967:		
conv2d_32_52969:	)
conv2d_33_52983:		
conv2d_33_52985:	)
conv2d_34_52999:		
conv2d_34_53001:	
identity¢!conv2d_27/StatefulPartitionedCall¢!conv2d_28/StatefulPartitionedCall¢!conv2d_29/StatefulPartitionedCall¢!conv2d_30/StatefulPartitionedCall¢!conv2d_31/StatefulPartitionedCall¢!conv2d_32/StatefulPartitionedCall¢!conv2d_33/StatefulPartitionedCall¢!conv2d_34/StatefulPartitionedCall
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_52887conv2d_27_52889*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:~~	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_528862#
!conv2d_27/StatefulPartitionedCall¹
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_52903conv2d_28_52905*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_529022#
!conv2d_28/StatefulPartitionedCall¹
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_52919conv2d_29_52921*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_529182#
!conv2d_29/StatefulPartitionedCall¹
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_52935conv2d_30_52937*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_529342#
!conv2d_30/StatefulPartitionedCall¹
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0conv2d_31_52951conv2d_31_52953*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:vv	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_529502#
!conv2d_31/StatefulPartitionedCall¹
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_52967conv2d_32_52969*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_529662#
!conv2d_32/StatefulPartitionedCall¹
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0conv2d_33_52983conv2d_33_52985*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:rr	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_529822#
!conv2d_33/StatefulPartitionedCall¹
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_52999conv2d_34_53001*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_529982#
!conv2d_34/StatefulPartitionedCall
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs


ý
D__inference_conv2d_30_layer_call_and_return_conditional_losses_53647

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:xx	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:zz	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:zz	
 
_user_specified_nameinputs
Ì
â
5__inference_small_conv2d_stacked8_layer_call_fn_53040
input_1!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCall»
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
:pp	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_530052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ý
D__inference_conv2d_31_layer_call_and_return_conditional_losses_52950

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:vv	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:xx	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:xx	
 
_user_specified_nameinputs

Ð
#__inference_signature_wrapper_53402
input_1!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:		
	unknown_4:	#
	unknown_5:		
	unknown_6:	#
	unknown_7:		
	unknown_8:	#
	unknown_9:		

unknown_10:	$

unknown_11:		

unknown_12:	$

unknown_13:		

unknown_14:	
identity¢StatefulPartitionedCall
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
:pp	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_528692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ý
D__inference_conv2d_27_layer_call_and_return_conditional_losses_53590

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:~~	2

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
å,
à
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53319
input_1)
conv2d_27_53278:	
conv2d_27_53280:	)
conv2d_28_53283:		
conv2d_28_53285:	)
conv2d_29_53288:		
conv2d_29_53290:	)
conv2d_30_53293:		
conv2d_30_53295:	)
conv2d_31_53298:		
conv2d_31_53300:	)
conv2d_32_53303:		
conv2d_32_53305:	)
conv2d_33_53308:		
conv2d_33_53310:	)
conv2d_34_53313:		
conv2d_34_53315:	
identity¢!conv2d_27/StatefulPartitionedCall¢!conv2d_28/StatefulPartitionedCall¢!conv2d_29/StatefulPartitionedCall¢!conv2d_30/StatefulPartitionedCall¢!conv2d_31/StatefulPartitionedCall¢!conv2d_32/StatefulPartitionedCall¢!conv2d_33/StatefulPartitionedCall¢!conv2d_34/StatefulPartitionedCall
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_27_53278conv2d_27_53280*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:~~	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_528862#
!conv2d_27/StatefulPartitionedCall¹
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_53283conv2d_28_53285*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_529022#
!conv2d_28/StatefulPartitionedCall¹
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_53288conv2d_29_53290*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_529182#
!conv2d_29/StatefulPartitionedCall¹
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_53293conv2d_30_53295*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_529342#
!conv2d_30/StatefulPartitionedCall¹
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0conv2d_31_53298conv2d_31_53300*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:vv	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_529502#
!conv2d_31/StatefulPartitionedCall¹
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_53303conv2d_32_53305*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_529662#
!conv2d_32/StatefulPartitionedCall¹
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0conv2d_33_53308conv2d_33_53310*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:rr	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_529822#
!conv2d_33/StatefulPartitionedCall¹
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_53313conv2d_34_53315*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_529982#
!conv2d_34/StatefulPartitionedCall
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


ý
D__inference_conv2d_32_layer_call_and_return_conditional_losses_53685

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:vv	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:vv	
 
_user_specified_nameinputs


ý
D__inference_conv2d_32_layer_call_and_return_conditional_losses_52966

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:tt	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:vv	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:vv	
 
_user_specified_nameinputs


ý
D__inference_conv2d_30_layer_call_and_return_conditional_losses_52934

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:xx	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:zz	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:zz	
 
_user_specified_nameinputs
õj

 __inference__wrapped_model_52869
input_1X
>small_conv2d_stacked8_conv2d_27_conv2d_readvariableop_resource:	M
?small_conv2d_stacked8_conv2d_27_biasadd_readvariableop_resource:	X
>small_conv2d_stacked8_conv2d_28_conv2d_readvariableop_resource:		M
?small_conv2d_stacked8_conv2d_28_biasadd_readvariableop_resource:	X
>small_conv2d_stacked8_conv2d_29_conv2d_readvariableop_resource:		M
?small_conv2d_stacked8_conv2d_29_biasadd_readvariableop_resource:	X
>small_conv2d_stacked8_conv2d_30_conv2d_readvariableop_resource:		M
?small_conv2d_stacked8_conv2d_30_biasadd_readvariableop_resource:	X
>small_conv2d_stacked8_conv2d_31_conv2d_readvariableop_resource:		M
?small_conv2d_stacked8_conv2d_31_biasadd_readvariableop_resource:	X
>small_conv2d_stacked8_conv2d_32_conv2d_readvariableop_resource:		M
?small_conv2d_stacked8_conv2d_32_biasadd_readvariableop_resource:	X
>small_conv2d_stacked8_conv2d_33_conv2d_readvariableop_resource:		M
?small_conv2d_stacked8_conv2d_33_biasadd_readvariableop_resource:	X
>small_conv2d_stacked8_conv2d_34_conv2d_readvariableop_resource:		M
?small_conv2d_stacked8_conv2d_34_biasadd_readvariableop_resource:	
identity¢6small_conv2d_stacked8/conv2d_27/BiasAdd/ReadVariableOp¢5small_conv2d_stacked8/conv2d_27/Conv2D/ReadVariableOp¢6small_conv2d_stacked8/conv2d_28/BiasAdd/ReadVariableOp¢5small_conv2d_stacked8/conv2d_28/Conv2D/ReadVariableOp¢6small_conv2d_stacked8/conv2d_29/BiasAdd/ReadVariableOp¢5small_conv2d_stacked8/conv2d_29/Conv2D/ReadVariableOp¢6small_conv2d_stacked8/conv2d_30/BiasAdd/ReadVariableOp¢5small_conv2d_stacked8/conv2d_30/Conv2D/ReadVariableOp¢6small_conv2d_stacked8/conv2d_31/BiasAdd/ReadVariableOp¢5small_conv2d_stacked8/conv2d_31/Conv2D/ReadVariableOp¢6small_conv2d_stacked8/conv2d_32/BiasAdd/ReadVariableOp¢5small_conv2d_stacked8/conv2d_32/Conv2D/ReadVariableOp¢6small_conv2d_stacked8/conv2d_33/BiasAdd/ReadVariableOp¢5small_conv2d_stacked8/conv2d_33/Conv2D/ReadVariableOp¢6small_conv2d_stacked8/conv2d_34/BiasAdd/ReadVariableOp¢5small_conv2d_stacked8/conv2d_34/Conv2D/ReadVariableOpõ
5small_conv2d_stacked8/conv2d_27/Conv2D/ReadVariableOpReadVariableOp>small_conv2d_stacked8_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype027
5small_conv2d_stacked8/conv2d_27/Conv2D/ReadVariableOpü
&small_conv2d_stacked8/conv2d_27/Conv2DConv2Dinput_1=small_conv2d_stacked8/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	*
paddingVALID*
strides
2(
&small_conv2d_stacked8/conv2d_27/Conv2Dì
6small_conv2d_stacked8/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp?small_conv2d_stacked8_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype028
6small_conv2d_stacked8/conv2d_27/BiasAdd/ReadVariableOpÿ
'small_conv2d_stacked8/conv2d_27/BiasAddBiasAdd/small_conv2d_stacked8/conv2d_27/Conv2D:output:0>small_conv2d_stacked8/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	2)
'small_conv2d_stacked8/conv2d_27/BiasAddõ
5small_conv2d_stacked8/conv2d_28/Conv2D/ReadVariableOpReadVariableOp>small_conv2d_stacked8_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype027
5small_conv2d_stacked8/conv2d_28/Conv2D/ReadVariableOp¥
&small_conv2d_stacked8/conv2d_28/Conv2DConv2D0small_conv2d_stacked8/conv2d_27/BiasAdd:output:0=small_conv2d_stacked8/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	*
paddingVALID*
strides
2(
&small_conv2d_stacked8/conv2d_28/Conv2Dì
6small_conv2d_stacked8/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp?small_conv2d_stacked8_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype028
6small_conv2d_stacked8/conv2d_28/BiasAdd/ReadVariableOpÿ
'small_conv2d_stacked8/conv2d_28/BiasAddBiasAdd/small_conv2d_stacked8/conv2d_28/Conv2D:output:0>small_conv2d_stacked8/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	2)
'small_conv2d_stacked8/conv2d_28/BiasAddõ
5small_conv2d_stacked8/conv2d_29/Conv2D/ReadVariableOpReadVariableOp>small_conv2d_stacked8_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype027
5small_conv2d_stacked8/conv2d_29/Conv2D/ReadVariableOp¥
&small_conv2d_stacked8/conv2d_29/Conv2DConv2D0small_conv2d_stacked8/conv2d_28/BiasAdd:output:0=small_conv2d_stacked8/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2(
&small_conv2d_stacked8/conv2d_29/Conv2Dì
6small_conv2d_stacked8/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp?small_conv2d_stacked8_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype028
6small_conv2d_stacked8/conv2d_29/BiasAdd/ReadVariableOpÿ
'small_conv2d_stacked8/conv2d_29/BiasAddBiasAdd/small_conv2d_stacked8/conv2d_29/Conv2D:output:0>small_conv2d_stacked8/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2)
'small_conv2d_stacked8/conv2d_29/BiasAddõ
5small_conv2d_stacked8/conv2d_30/Conv2D/ReadVariableOpReadVariableOp>small_conv2d_stacked8_conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype027
5small_conv2d_stacked8/conv2d_30/Conv2D/ReadVariableOp¥
&small_conv2d_stacked8/conv2d_30/Conv2DConv2D0small_conv2d_stacked8/conv2d_29/BiasAdd:output:0=small_conv2d_stacked8/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	*
paddingVALID*
strides
2(
&small_conv2d_stacked8/conv2d_30/Conv2Dì
6small_conv2d_stacked8/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp?small_conv2d_stacked8_conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype028
6small_conv2d_stacked8/conv2d_30/BiasAdd/ReadVariableOpÿ
'small_conv2d_stacked8/conv2d_30/BiasAddBiasAdd/small_conv2d_stacked8/conv2d_30/Conv2D:output:0>small_conv2d_stacked8/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	2)
'small_conv2d_stacked8/conv2d_30/BiasAddõ
5small_conv2d_stacked8/conv2d_31/Conv2D/ReadVariableOpReadVariableOp>small_conv2d_stacked8_conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype027
5small_conv2d_stacked8/conv2d_31/Conv2D/ReadVariableOp¥
&small_conv2d_stacked8/conv2d_31/Conv2DConv2D0small_conv2d_stacked8/conv2d_30/BiasAdd:output:0=small_conv2d_stacked8/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	*
paddingVALID*
strides
2(
&small_conv2d_stacked8/conv2d_31/Conv2Dì
6small_conv2d_stacked8/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp?small_conv2d_stacked8_conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype028
6small_conv2d_stacked8/conv2d_31/BiasAdd/ReadVariableOpÿ
'small_conv2d_stacked8/conv2d_31/BiasAddBiasAdd/small_conv2d_stacked8/conv2d_31/Conv2D:output:0>small_conv2d_stacked8/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	2)
'small_conv2d_stacked8/conv2d_31/BiasAddõ
5small_conv2d_stacked8/conv2d_32/Conv2D/ReadVariableOpReadVariableOp>small_conv2d_stacked8_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype027
5small_conv2d_stacked8/conv2d_32/Conv2D/ReadVariableOp¥
&small_conv2d_stacked8/conv2d_32/Conv2DConv2D0small_conv2d_stacked8/conv2d_31/BiasAdd:output:0=small_conv2d_stacked8/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2(
&small_conv2d_stacked8/conv2d_32/Conv2Dì
6small_conv2d_stacked8/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp?small_conv2d_stacked8_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype028
6small_conv2d_stacked8/conv2d_32/BiasAdd/ReadVariableOpÿ
'small_conv2d_stacked8/conv2d_32/BiasAddBiasAdd/small_conv2d_stacked8/conv2d_32/Conv2D:output:0>small_conv2d_stacked8/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2)
'small_conv2d_stacked8/conv2d_32/BiasAddõ
5small_conv2d_stacked8/conv2d_33/Conv2D/ReadVariableOpReadVariableOp>small_conv2d_stacked8_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype027
5small_conv2d_stacked8/conv2d_33/Conv2D/ReadVariableOp¥
&small_conv2d_stacked8/conv2d_33/Conv2DConv2D0small_conv2d_stacked8/conv2d_32/BiasAdd:output:0=small_conv2d_stacked8/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	*
paddingVALID*
strides
2(
&small_conv2d_stacked8/conv2d_33/Conv2Dì
6small_conv2d_stacked8/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp?small_conv2d_stacked8_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype028
6small_conv2d_stacked8/conv2d_33/BiasAdd/ReadVariableOpÿ
'small_conv2d_stacked8/conv2d_33/BiasAddBiasAdd/small_conv2d_stacked8/conv2d_33/Conv2D:output:0>small_conv2d_stacked8/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	2)
'small_conv2d_stacked8/conv2d_33/BiasAddõ
5small_conv2d_stacked8/conv2d_34/Conv2D/ReadVariableOpReadVariableOp>small_conv2d_stacked8_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype027
5small_conv2d_stacked8/conv2d_34/Conv2D/ReadVariableOp¥
&small_conv2d_stacked8/conv2d_34/Conv2DConv2D0small_conv2d_stacked8/conv2d_33/BiasAdd:output:0=small_conv2d_stacked8/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	*
paddingVALID*
strides
2(
&small_conv2d_stacked8/conv2d_34/Conv2Dì
6small_conv2d_stacked8/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp?small_conv2d_stacked8_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype028
6small_conv2d_stacked8/conv2d_34/BiasAdd/ReadVariableOpÿ
'small_conv2d_stacked8/conv2d_34/BiasAddBiasAdd/small_conv2d_stacked8/conv2d_34/Conv2D:output:0>small_conv2d_stacked8/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	2)
'small_conv2d_stacked8/conv2d_34/BiasAdd
IdentityIdentity0small_conv2d_stacked8/conv2d_34/BiasAdd:output:07^small_conv2d_stacked8/conv2d_27/BiasAdd/ReadVariableOp6^small_conv2d_stacked8/conv2d_27/Conv2D/ReadVariableOp7^small_conv2d_stacked8/conv2d_28/BiasAdd/ReadVariableOp6^small_conv2d_stacked8/conv2d_28/Conv2D/ReadVariableOp7^small_conv2d_stacked8/conv2d_29/BiasAdd/ReadVariableOp6^small_conv2d_stacked8/conv2d_29/Conv2D/ReadVariableOp7^small_conv2d_stacked8/conv2d_30/BiasAdd/ReadVariableOp6^small_conv2d_stacked8/conv2d_30/Conv2D/ReadVariableOp7^small_conv2d_stacked8/conv2d_31/BiasAdd/ReadVariableOp6^small_conv2d_stacked8/conv2d_31/Conv2D/ReadVariableOp7^small_conv2d_stacked8/conv2d_32/BiasAdd/ReadVariableOp6^small_conv2d_stacked8/conv2d_32/Conv2D/ReadVariableOp7^small_conv2d_stacked8/conv2d_33/BiasAdd/ReadVariableOp6^small_conv2d_stacked8/conv2d_33/Conv2D/ReadVariableOp7^small_conv2d_stacked8/conv2d_34/BiasAdd/ReadVariableOp6^small_conv2d_stacked8/conv2d_34/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2p
6small_conv2d_stacked8/conv2d_27/BiasAdd/ReadVariableOp6small_conv2d_stacked8/conv2d_27/BiasAdd/ReadVariableOp2n
5small_conv2d_stacked8/conv2d_27/Conv2D/ReadVariableOp5small_conv2d_stacked8/conv2d_27/Conv2D/ReadVariableOp2p
6small_conv2d_stacked8/conv2d_28/BiasAdd/ReadVariableOp6small_conv2d_stacked8/conv2d_28/BiasAdd/ReadVariableOp2n
5small_conv2d_stacked8/conv2d_28/Conv2D/ReadVariableOp5small_conv2d_stacked8/conv2d_28/Conv2D/ReadVariableOp2p
6small_conv2d_stacked8/conv2d_29/BiasAdd/ReadVariableOp6small_conv2d_stacked8/conv2d_29/BiasAdd/ReadVariableOp2n
5small_conv2d_stacked8/conv2d_29/Conv2D/ReadVariableOp5small_conv2d_stacked8/conv2d_29/Conv2D/ReadVariableOp2p
6small_conv2d_stacked8/conv2d_30/BiasAdd/ReadVariableOp6small_conv2d_stacked8/conv2d_30/BiasAdd/ReadVariableOp2n
5small_conv2d_stacked8/conv2d_30/Conv2D/ReadVariableOp5small_conv2d_stacked8/conv2d_30/Conv2D/ReadVariableOp2p
6small_conv2d_stacked8/conv2d_31/BiasAdd/ReadVariableOp6small_conv2d_stacked8/conv2d_31/BiasAdd/ReadVariableOp2n
5small_conv2d_stacked8/conv2d_31/Conv2D/ReadVariableOp5small_conv2d_stacked8/conv2d_31/Conv2D/ReadVariableOp2p
6small_conv2d_stacked8/conv2d_32/BiasAdd/ReadVariableOp6small_conv2d_stacked8/conv2d_32/BiasAdd/ReadVariableOp2n
5small_conv2d_stacked8/conv2d_32/Conv2D/ReadVariableOp5small_conv2d_stacked8/conv2d_32/Conv2D/ReadVariableOp2p
6small_conv2d_stacked8/conv2d_33/BiasAdd/ReadVariableOp6small_conv2d_stacked8/conv2d_33/BiasAdd/ReadVariableOp2n
5small_conv2d_stacked8/conv2d_33/Conv2D/ReadVariableOp5small_conv2d_stacked8/conv2d_33/Conv2D/ReadVariableOp2p
6small_conv2d_stacked8/conv2d_34/BiasAdd/ReadVariableOp6small_conv2d_stacked8/conv2d_34/BiasAdd/ReadVariableOp2n
5small_conv2d_stacked8/conv2d_34/Conv2D/ReadVariableOp5small_conv2d_stacked8/conv2d_34/Conv2D/ReadVariableOp:Q M
(
_output_shapes
:
!
_user_specified_name	input_1


)__inference_conv2d_34_layer_call_fn_53732

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_529982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:rr	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:rr	
 
_user_specified_nameinputs


ý
D__inference_conv2d_29_layer_call_and_return_conditional_losses_53628

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:zz	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:||	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:||	
 
_user_specified_nameinputs


ý
D__inference_conv2d_28_layer_call_and_return_conditional_losses_52902

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:||	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:~~	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:~~	
 
_user_specified_nameinputs


ý
D__inference_conv2d_28_layer_call_and_return_conditional_losses_53609

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:||	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:~~	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:~~	
 
_user_specified_nameinputs
â,
ß
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53203

inputs)
conv2d_27_53162:	
conv2d_27_53164:	)
conv2d_28_53167:		
conv2d_28_53169:	)
conv2d_29_53172:		
conv2d_29_53174:	)
conv2d_30_53177:		
conv2d_30_53179:	)
conv2d_31_53182:		
conv2d_31_53184:	)
conv2d_32_53187:		
conv2d_32_53189:	)
conv2d_33_53192:		
conv2d_33_53194:	)
conv2d_34_53197:		
conv2d_34_53199:	
identity¢!conv2d_27/StatefulPartitionedCall¢!conv2d_28/StatefulPartitionedCall¢!conv2d_29/StatefulPartitionedCall¢!conv2d_30/StatefulPartitionedCall¢!conv2d_31/StatefulPartitionedCall¢!conv2d_32/StatefulPartitionedCall¢!conv2d_33/StatefulPartitionedCall¢!conv2d_34/StatefulPartitionedCall
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_53162conv2d_27_53164*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:~~	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_528862#
!conv2d_27/StatefulPartitionedCall¹
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_53167conv2d_28_53169*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:||	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_529022#
!conv2d_28/StatefulPartitionedCall¹
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_53172conv2d_29_53174*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:zz	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_529182#
!conv2d_29/StatefulPartitionedCall¹
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_53177conv2d_30_53179*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_529342#
!conv2d_30/StatefulPartitionedCall¹
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0conv2d_31_53182conv2d_31_53184*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:vv	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_529502#
!conv2d_31/StatefulPartitionedCall¹
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_53187conv2d_32_53189*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_529662#
!conv2d_32/StatefulPartitionedCall¹
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0conv2d_33_53192conv2d_33_53194*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:rr	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_529822#
!conv2d_33/StatefulPartitionedCall¹
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_53197conv2d_34_53199*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:pp	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_529982#
!conv2d_34/StatefulPartitionedCall
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:P L
(
_output_shapes
:
 
_user_specified_nameinputs
úF
 

!__inference__traced_restore_53861
file_prefix;
!assignvariableop_conv2d_27_kernel:	/
!assignvariableop_1_conv2d_27_bias:	=
#assignvariableop_2_conv2d_28_kernel:		/
!assignvariableop_3_conv2d_28_bias:	=
#assignvariableop_4_conv2d_29_kernel:		/
!assignvariableop_5_conv2d_29_bias:	=
#assignvariableop_6_conv2d_30_kernel:		/
!assignvariableop_7_conv2d_30_bias:	=
#assignvariableop_8_conv2d_31_kernel:		/
!assignvariableop_9_conv2d_31_bias:	>
$assignvariableop_10_conv2d_32_kernel:		0
"assignvariableop_11_conv2d_32_bias:	>
$assignvariableop_12_conv2d_33_kernel:		0
"assignvariableop_13_conv2d_33_bias:	>
$assignvariableop_14_conv2d_34_kernel:		0
"assignvariableop_15_conv2d_34_bias:	
identity_17¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
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

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_27_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_27_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_28_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_28_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_29_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_29_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_30_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_30_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¨
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_31_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_31_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_32_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_32_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¬
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_33_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ª
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_33_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¬
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_34_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ª
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_34_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¾
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16±
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
û+
ë
__inference__traced_save_53803
file_prefix/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop/
+savev2_conv2d_30_kernel_read_readvariableop-
)savev2_conv2d_30_bias_read_readvariableop/
+savev2_conv2d_31_kernel_read_readvariableop-
)savev2_conv2d_31_bias_read_readvariableop/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop
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
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesª
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop+savev2_conv2d_30_kernel_read_readvariableop)savev2_conv2d_30_bias_read_readvariableop+savev2_conv2d_31_kernel_read_readvariableop)savev2_conv2d_31_bias_read_readvariableop+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*Ù
_input_shapesÇ
Ä: :	:	:		:	:		:	:		:	:		:	:		:	:		:	:		:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:	: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,	(
&
_output_shapes
:		: 


_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:

_output_shapes
: 


ý
D__inference_conv2d_33_layer_call_and_return_conditional_losses_53704

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*&
_output_shapes
:rr	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:tt	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:tt	
 
_user_specified_nameinputs
¢J
ÿ
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53506

inputsB
(conv2d_27_conv2d_readvariableop_resource:	7
)conv2d_27_biasadd_readvariableop_resource:	B
(conv2d_28_conv2d_readvariableop_resource:		7
)conv2d_28_biasadd_readvariableop_resource:	B
(conv2d_29_conv2d_readvariableop_resource:		7
)conv2d_29_biasadd_readvariableop_resource:	B
(conv2d_30_conv2d_readvariableop_resource:		7
)conv2d_30_biasadd_readvariableop_resource:	B
(conv2d_31_conv2d_readvariableop_resource:		7
)conv2d_31_biasadd_readvariableop_resource:	B
(conv2d_32_conv2d_readvariableop_resource:		7
)conv2d_32_biasadd_readvariableop_resource:	B
(conv2d_33_conv2d_readvariableop_resource:		7
)conv2d_33_biasadd_readvariableop_resource:	B
(conv2d_34_conv2d_readvariableop_resource:		7
)conv2d_34_biasadd_readvariableop_resource:	
identity¢ conv2d_27/BiasAdd/ReadVariableOp¢conv2d_27/Conv2D/ReadVariableOp¢ conv2d_28/BiasAdd/ReadVariableOp¢conv2d_28/Conv2D/ReadVariableOp¢ conv2d_29/BiasAdd/ReadVariableOp¢conv2d_29/Conv2D/ReadVariableOp¢ conv2d_30/BiasAdd/ReadVariableOp¢conv2d_30/Conv2D/ReadVariableOp¢ conv2d_31/BiasAdd/ReadVariableOp¢conv2d_31/Conv2D/ReadVariableOp¢ conv2d_32/BiasAdd/ReadVariableOp¢conv2d_32/Conv2D/ReadVariableOp¢ conv2d_33/BiasAdd/ReadVariableOp¢conv2d_33/Conv2D/ReadVariableOp¢ conv2d_34/BiasAdd/ReadVariableOp¢conv2d_34/Conv2D/ReadVariableOp³
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02!
conv2d_27/Conv2D/ReadVariableOp¹
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	*
paddingVALID*
strides
2
conv2d_27/Conv2Dª
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp§
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:~~	2
conv2d_27/BiasAdd³
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_28/Conv2D/ReadVariableOpÍ
conv2d_28/Conv2DConv2Dconv2d_27/BiasAdd:output:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	*
paddingVALID*
strides
2
conv2d_28/Conv2Dª
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp§
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:||	2
conv2d_28/BiasAdd³
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_29/Conv2D/ReadVariableOpÍ
conv2d_29/Conv2DConv2Dconv2d_28/BiasAdd:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	*
paddingVALID*
strides
2
conv2d_29/Conv2Dª
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp§
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:zz	2
conv2d_29/BiasAdd³
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_30/Conv2D/ReadVariableOpÍ
conv2d_30/Conv2DConv2Dconv2d_29/BiasAdd:output:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	*
paddingVALID*
strides
2
conv2d_30/Conv2Dª
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp§
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:xx	2
conv2d_30/BiasAdd³
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_31/Conv2D/ReadVariableOpÍ
conv2d_31/Conv2DConv2Dconv2d_30/BiasAdd:output:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	*
paddingVALID*
strides
2
conv2d_31/Conv2Dª
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp§
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:vv	2
conv2d_31/BiasAdd³
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_32/Conv2D/ReadVariableOpÍ
conv2d_32/Conv2DConv2Dconv2d_31/BiasAdd:output:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	*
paddingVALID*
strides
2
conv2d_32/Conv2Dª
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp§
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:tt	2
conv2d_32/BiasAdd³
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_33/Conv2D/ReadVariableOpÍ
conv2d_33/Conv2DConv2Dconv2d_32/BiasAdd:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	*
paddingVALID*
strides
2
conv2d_33/Conv2Dª
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp§
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:rr	2
conv2d_33/BiasAdd³
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02!
conv2d_34/Conv2D/ReadVariableOpÍ
conv2d_34/Conv2DConv2Dconv2d_33/BiasAdd:output:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	*
paddingVALID*
strides
2
conv2d_34/Conv2Dª
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp§
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:pp	2
conv2d_34/BiasAdd
IdentityIdentityconv2d_34/BiasAdd:output:0!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:pp	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:: : : : : : : : : : : : : : : : 2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp:P L
(
_output_shapes
:
 
_user_specified_nameinputs


)__inference_conv2d_30_layer_call_fn_53656

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:xx	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_529342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:xx	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:zz	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:zz	
 
_user_specified_nameinputs


)__inference_conv2d_32_layer_call_fn_53694

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:tt	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_529662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:tt	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:vv	: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:vv	
 
_user_specified_nameinputs"ÌL
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
	conv2d_34/
StatefulPartitionedCall:0pp	tensorflow/serving/predict:åÅ
Ãq
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
n_default_save_signature"Âm
_tf_keras_network¦m{"name": "small_conv2d_stacked8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "small_conv2d_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_30", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_31", "inbound_nodes": [[["conv2d_30", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_32", "inbound_nodes": [[["conv2d_31", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_33", "inbound_nodes": [[["conv2d_32", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_34", "inbound_nodes": [[["conv2d_33", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_34", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "small_conv2d_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_30", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_31", "inbound_nodes": [[["conv2d_30", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Conv2D", "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_32", "inbound_nodes": [[["conv2d_31", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv2D", "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_33", "inbound_nodes": [[["conv2d_32", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_34", "inbound_nodes": [[["conv2d_33", 0, 0, {}]]], "shared_object_id": 24}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_34", 0, 0]]}}}
÷"ô
_tf_keras_input_layerÔ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ü


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*o&call_and_return_all_conditional_losses
p__call__"×	
_tf_keras_layer½	{"name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 3]}}
þ


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*q&call_and_return_all_conditional_losses
r__call__"Ù	
_tf_keras_layer¿	{"name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_27", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 126, 126, 9]}}
þ


kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
*s&call_and_return_all_conditional_losses
t__call__"Ù	
_tf_keras_layer¿	{"name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_28", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 124, 124, 9]}}


!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
*u&call_and_return_all_conditional_losses
v__call__"Ü	
_tf_keras_layerÂ	{"name": "conv2d_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_29", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 122, 122, 9]}}


'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Ü	
_tf_keras_layerÂ	{"name": "conv2d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_30", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 120, 120, 9]}}


-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
*y&call_and_return_all_conditional_losses
z__call__"Ü	
_tf_keras_layerÂ	{"name": "conv2d_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_31", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 118, 118, 9]}}


3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
*{&call_and_return_all_conditional_losses
|__call__"Ü	
_tf_keras_layerÂ	{"name": "conv2d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_32", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 116, 116, 9]}}


9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
*}&call_and_return_all_conditional_losses
~__call__"Ü	
_tf_keras_layerÂ	{"name": "conv2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_33", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 9}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 114, 114, 9]}}

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

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
Ê

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
*:(	2conv2d_27/kernel
:	2conv2d_27/bias
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
­

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
*:(		2conv2d_28/kernel
:	2conv2d_28/bias
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
­

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
*:(		2conv2d_29/kernel
:	2conv2d_29/bias
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
­

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
*:(		2conv2d_30/kernel
:	2conv2d_30/bias
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
­

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
*:(		2conv2d_31/kernel
:	2conv2d_31/bias
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
­

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
*:(		2conv2d_32/kernel
:	2conv2d_32/bias
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
­

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
*:(		2conv2d_33/kernel
:	2conv2d_33/bias
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
­

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
*:(		2conv2d_34/kernel
:	2conv2d_34/bias
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
­

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
2
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53454
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53506
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53319
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53363À
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
¢2
5__inference_small_conv2d_stacked8_layer_call_fn_53040
5__inference_small_conv2d_stacked8_layer_call_fn_53543
5__inference_small_conv2d_stacked8_layer_call_fn_53580
5__inference_small_conv2d_stacked8_layer_call_fn_53275À
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
 __inference__wrapped_model_52869·
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
D__inference_conv2d_27_layer_call_and_return_conditional_losses_53590¢
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
)__inference_conv2d_27_layer_call_fn_53599¢
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
D__inference_conv2d_28_layer_call_and_return_conditional_losses_53609¢
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
)__inference_conv2d_28_layer_call_fn_53618¢
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
D__inference_conv2d_29_layer_call_and_return_conditional_losses_53628¢
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
)__inference_conv2d_29_layer_call_fn_53637¢
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
D__inference_conv2d_30_layer_call_and_return_conditional_losses_53647¢
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
)__inference_conv2d_30_layer_call_fn_53656¢
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
D__inference_conv2d_31_layer_call_and_return_conditional_losses_53666¢
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
)__inference_conv2d_31_layer_call_fn_53675¢
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
D__inference_conv2d_32_layer_call_and_return_conditional_losses_53685¢
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
)__inference_conv2d_32_layer_call_fn_53694¢
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
D__inference_conv2d_33_layer_call_and_return_conditional_losses_53704¢
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
)__inference_conv2d_33_layer_call_fn_53713¢
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
D__inference_conv2d_34_layer_call_and_return_conditional_losses_53723¢
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
)__inference_conv2d_34_layer_call_fn_53732¢
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
#__inference_signature_wrapper_53402input_1"
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
 
 __inference__wrapped_model_52869{!"'(-.349:1¢.
'¢$
"
input_1
ª "4ª1
/
	conv2d_34"
	conv2d_34pp	¤
D__inference_conv2d_27_layer_call_and_return_conditional_losses_53590\0¢-
&¢#
!
inputs
ª "$¢!

0~~	
 |
)__inference_conv2d_27_layer_call_fn_53599O0¢-
&¢#
!
inputs
ª "~~	¢
D__inference_conv2d_28_layer_call_and_return_conditional_losses_53609Z.¢+
$¢!

inputs~~	
ª "$¢!

0||	
 z
)__inference_conv2d_28_layer_call_fn_53618M.¢+
$¢!

inputs~~	
ª "||	¢
D__inference_conv2d_29_layer_call_and_return_conditional_losses_53628Z.¢+
$¢!

inputs||	
ª "$¢!

0zz	
 z
)__inference_conv2d_29_layer_call_fn_53637M.¢+
$¢!

inputs||	
ª "zz	¢
D__inference_conv2d_30_layer_call_and_return_conditional_losses_53647Z!".¢+
$¢!

inputszz	
ª "$¢!

0xx	
 z
)__inference_conv2d_30_layer_call_fn_53656M!".¢+
$¢!

inputszz	
ª "xx	¢
D__inference_conv2d_31_layer_call_and_return_conditional_losses_53666Z'(.¢+
$¢!

inputsxx	
ª "$¢!

0vv	
 z
)__inference_conv2d_31_layer_call_fn_53675M'(.¢+
$¢!

inputsxx	
ª "vv	¢
D__inference_conv2d_32_layer_call_and_return_conditional_losses_53685Z-..¢+
$¢!

inputsvv	
ª "$¢!

0tt	
 z
)__inference_conv2d_32_layer_call_fn_53694M-..¢+
$¢!

inputsvv	
ª "tt	¢
D__inference_conv2d_33_layer_call_and_return_conditional_losses_53704Z34.¢+
$¢!

inputstt	
ª "$¢!

0rr	
 z
)__inference_conv2d_33_layer_call_fn_53713M34.¢+
$¢!

inputstt	
ª "rr	¢
D__inference_conv2d_34_layer_call_and_return_conditional_losses_53723Z9:.¢+
$¢!

inputsrr	
ª "$¢!

0pp	
 z
)__inference_conv2d_34_layer_call_fn_53732M9:.¢+
$¢!

inputsrr	
ª "pp	®
#__inference_signature_wrapper_53402!"'(-.349:<¢9
¢ 
2ª/
-
input_1"
input_1"4ª1
/
	conv2d_34"
	conv2d_34pp	Ç
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53319s!"'(-.349:9¢6
/¢,
"
input_1
p 

 
ª "$¢!

0pp	
 Ç
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53363s!"'(-.349:9¢6
/¢,
"
input_1
p

 
ª "$¢!

0pp	
 Æ
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53454r!"'(-.349:8¢5
.¢+
!
inputs
p 

 
ª "$¢!

0pp	
 Æ
P__inference_small_conv2d_stacked8_layer_call_and_return_conditional_losses_53506r!"'(-.349:8¢5
.¢+
!
inputs
p

 
ª "$¢!

0pp	
 
5__inference_small_conv2d_stacked8_layer_call_fn_53040f!"'(-.349:9¢6
/¢,
"
input_1
p 

 
ª "pp	
5__inference_small_conv2d_stacked8_layer_call_fn_53275f!"'(-.349:9¢6
/¢,
"
input_1
p

 
ª "pp	
5__inference_small_conv2d_stacked8_layer_call_fn_53543e!"'(-.349:8¢5
.¢+
!
inputs
p 

 
ª "pp	
5__inference_small_conv2d_stacked8_layer_call_fn_53580e!"'(-.349:8¢5
.¢+
!
inputs
p

 
ª "pp	