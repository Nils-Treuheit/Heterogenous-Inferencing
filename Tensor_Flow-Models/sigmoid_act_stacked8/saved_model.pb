ª
Â


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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
	separatorstring "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ŞÄ

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

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
GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_10102
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU 2J 8 *'
f"R 
__inference__traced_save_10239

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
GPU 2J 8 **
f%R#
!__inference__traced_restore_10249â¦
Â
d
H__inference_activation_22_layer_call_and_return_conditional_losses_10201

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Æ
<
__inference__wrapped_model_9929
input_2
identity
*sigmoid_act_stacked8/activation_16/SigmoidSigmoidinput_2*
T0*
_output_shapes
:	2,
*sigmoid_act_stacked8/activation_16/Sigmoid½
*sigmoid_act_stacked8/activation_17/SigmoidSigmoid.sigmoid_act_stacked8/activation_16/Sigmoid:y:0*
T0*
_output_shapes
:	2,
*sigmoid_act_stacked8/activation_17/Sigmoid½
*sigmoid_act_stacked8/activation_18/SigmoidSigmoid.sigmoid_act_stacked8/activation_17/Sigmoid:y:0*
T0*
_output_shapes
:	2,
*sigmoid_act_stacked8/activation_18/Sigmoid½
*sigmoid_act_stacked8/activation_19/SigmoidSigmoid.sigmoid_act_stacked8/activation_18/Sigmoid:y:0*
T0*
_output_shapes
:	2,
*sigmoid_act_stacked8/activation_19/Sigmoid½
*sigmoid_act_stacked8/activation_20/SigmoidSigmoid.sigmoid_act_stacked8/activation_19/Sigmoid:y:0*
T0*
_output_shapes
:	2,
*sigmoid_act_stacked8/activation_20/Sigmoid½
*sigmoid_act_stacked8/activation_21/SigmoidSigmoid.sigmoid_act_stacked8/activation_20/Sigmoid:y:0*
T0*
_output_shapes
:	2,
*sigmoid_act_stacked8/activation_21/Sigmoid½
*sigmoid_act_stacked8/activation_22/SigmoidSigmoid.sigmoid_act_stacked8/activation_21/Sigmoid:y:0*
T0*
_output_shapes
:	2,
*sigmoid_act_stacked8/activation_22/Sigmoid½
*sigmoid_act_stacked8/activation_23/SigmoidSigmoid.sigmoid_act_stacked8/activation_22/Sigmoid:y:0*
T0*
_output_shapes
:	2,
*sigmoid_act_stacked8/activation_23/Sigmoidz
IdentityIdentity.sigmoid_act_stacked8/activation_23/Sigmoid:y:0*
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
±
P
4__inference_sigmoid_act_stacked8_layer_call_fn_10131

inputs
identityÄ
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
GPU 2J 8 *W
fRRP
N__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_99932
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
Â
d
H__inference_activation_20_layer_call_and_return_conditional_losses_10181

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Á
c
G__inference_activation_17_layer_call_and_return_conditional_losses_9948

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
£
I
-__inference_activation_22_layer_call_fn_10206

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
G__inference_activation_22_layer_call_and_return_conditional_losses_99832
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
Â
d
H__inference_activation_17_layer_call_and_return_conditional_losses_10151

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
£
I
-__inference_activation_16_layer_call_fn_10146

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
G__inference_activation_16_layer_call_and_return_conditional_losses_99412
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
Â
d
H__inference_activation_19_layer_call_and_return_conditional_losses_10171

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Ò
j
N__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_9993

inputs
identityÙ
activation_16/PartitionedCallPartitionedCallinputs*
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
G__inference_activation_16_layer_call_and_return_conditional_losses_99412
activation_16/PartitionedCallù
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
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
G__inference_activation_17_layer_call_and_return_conditional_losses_99482
activation_17/PartitionedCallù
activation_18/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
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
G__inference_activation_18_layer_call_and_return_conditional_losses_99552
activation_18/PartitionedCallù
activation_19/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
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
G__inference_activation_19_layer_call_and_return_conditional_losses_99622
activation_19/PartitionedCallù
activation_20/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
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
G__inference_activation_20_layer_call_and_return_conditional_losses_99692
activation_20/PartitionedCallù
activation_21/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
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
G__inference_activation_21_layer_call_and_return_conditional_losses_99762
activation_21/PartitionedCallù
activation_22/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
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
G__inference_activation_22_layer_call_and_return_conditional_losses_99832
activation_22/PartitionedCallù
activation_23/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
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
G__inference_activation_23_layer_call_and_return_conditional_losses_99902
activation_23/PartitionedCallr
IdentityIdentity&activation_23/PartitionedCall:output:0*
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
²
P
4__inference_sigmoid_act_stacked8_layer_call_fn_10136

inputs
identityÅ
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
GPU 2J 8 *X
fSRQ
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_100632
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
Â
d
H__inference_activation_18_layer_call_and_return_conditional_losses_10161

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
³
P
3__inference_sigmoid_act_stacked8_layer_call_fn_9996
input_2
identityÅ
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
GPU 2J 8 *W
fRRP
N__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_99932
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
£
I
-__inference_activation_20_layer_call_fn_10186

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
G__inference_activation_20_layer_call_and_return_conditional_losses_99692
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
Á
c
G__inference_activation_22_layer_call_and_return_conditional_losses_9983

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Á
c
G__inference_activation_16_layer_call_and_return_conditional_losses_9941

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Ö
l
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10083
input_2
identityÚ
activation_16/PartitionedCallPartitionedCallinput_2*
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
G__inference_activation_16_layer_call_and_return_conditional_losses_99412
activation_16/PartitionedCallù
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
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
G__inference_activation_17_layer_call_and_return_conditional_losses_99482
activation_17/PartitionedCallù
activation_18/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
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
G__inference_activation_18_layer_call_and_return_conditional_losses_99552
activation_18/PartitionedCallù
activation_19/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
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
G__inference_activation_19_layer_call_and_return_conditional_losses_99622
activation_19/PartitionedCallù
activation_20/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
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
G__inference_activation_20_layer_call_and_return_conditional_losses_99692
activation_20/PartitionedCallù
activation_21/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
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
G__inference_activation_21_layer_call_and_return_conditional_losses_99762
activation_21/PartitionedCallù
activation_22/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
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
G__inference_activation_22_layer_call_and_return_conditional_losses_99832
activation_22/PartitionedCallù
activation_23/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
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
G__inference_activation_23_layer_call_and_return_conditional_losses_99902
activation_23/PartitionedCallr
IdentityIdentity&activation_23/PartitionedCall:output:0*
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
µ
Q
4__inference_sigmoid_act_stacked8_layer_call_fn_10071
input_2
identityÆ
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
GPU 2J 8 *X
fSRQ
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_100632
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
ó

k
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10126

inputs
identityk
activation_16/SigmoidSigmoidinputs*
T0*
_output_shapes
:	2
activation_16/Sigmoid~
activation_17/SigmoidSigmoidactivation_16/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_17/Sigmoid~
activation_18/SigmoidSigmoidactivation_17/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_18/Sigmoid~
activation_19/SigmoidSigmoidactivation_18/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_19/Sigmoid~
activation_20/SigmoidSigmoidactivation_19/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_20/Sigmoid~
activation_21/SigmoidSigmoidactivation_20/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_21/Sigmoid~
activation_22/SigmoidSigmoidactivation_21/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_22/Sigmoid~
activation_23/SigmoidSigmoidactivation_22/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_23/Sigmoide
IdentityIdentityactivation_23/Sigmoid:y:0*
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
£
I
-__inference_activation_19_layer_call_fn_10176

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
G__inference_activation_19_layer_call_and_return_conditional_losses_99622
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
£
I
-__inference_activation_21_layer_call_fn_10196

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
G__inference_activation_21_layer_call_and_return_conditional_losses_99762
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
£
I
-__inference_activation_17_layer_call_fn_10156

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
G__inference_activation_17_layer_call_and_return_conditional_losses_99482
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
Á
c
G__inference_activation_19_layer_call_and_return_conditional_losses_9962

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Á
c
G__inference_activation_23_layer_call_and_return_conditional_losses_9990

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Â
d
H__inference_activation_21_layer_call_and_return_conditional_losses_10191

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Â
d
H__inference_activation_16_layer_call_and_return_conditional_losses_10141

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
£
I
-__inference_activation_18_layer_call_fn_10166

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
G__inference_activation_18_layer_call_and_return_conditional_losses_99552
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
Ö
l
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10095
input_2
identityÚ
activation_16/PartitionedCallPartitionedCallinput_2*
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
G__inference_activation_16_layer_call_and_return_conditional_losses_99412
activation_16/PartitionedCallù
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
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
G__inference_activation_17_layer_call_and_return_conditional_losses_99482
activation_17/PartitionedCallù
activation_18/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
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
G__inference_activation_18_layer_call_and_return_conditional_losses_99552
activation_18/PartitionedCallù
activation_19/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
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
G__inference_activation_19_layer_call_and_return_conditional_losses_99622
activation_19/PartitionedCallù
activation_20/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
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
G__inference_activation_20_layer_call_and_return_conditional_losses_99692
activation_20/PartitionedCallù
activation_21/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
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
G__inference_activation_21_layer_call_and_return_conditional_losses_99762
activation_21/PartitionedCallù
activation_22/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
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
G__inference_activation_22_layer_call_and_return_conditional_losses_99832
activation_22/PartitionedCallù
activation_23/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
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
G__inference_activation_23_layer_call_and_return_conditional_losses_99902
activation_23/PartitionedCallr
IdentityIdentity&activation_23/PartitionedCall:output:0*
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
£
I
-__inference_activation_23_layer_call_fn_10216

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
G__inference_activation_23_layer_call_and_return_conditional_losses_99902
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
Á
c
G__inference_activation_21_layer_call_and_return_conditional_losses_9976

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Â
d
H__inference_activation_23_layer_call_and_return_conditional_losses_10211

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Ô
k
__inference__traced_save_10239
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
ô
@
#__inference_signature_wrapper_10102
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
__inference__wrapped_model_99292
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
°
G
!__inference__traced_restore_10249
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
ó

k
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10114

inputs
identityk
activation_16/SigmoidSigmoidinputs*
T0*
_output_shapes
:	2
activation_16/Sigmoid~
activation_17/SigmoidSigmoidactivation_16/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_17/Sigmoid~
activation_18/SigmoidSigmoidactivation_17/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_18/Sigmoid~
activation_19/SigmoidSigmoidactivation_18/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_19/Sigmoid~
activation_20/SigmoidSigmoidactivation_19/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_20/Sigmoid~
activation_21/SigmoidSigmoidactivation_20/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_21/Sigmoid~
activation_22/SigmoidSigmoidactivation_21/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_22/Sigmoid~
activation_23/SigmoidSigmoidactivation_22/Sigmoid:y:0*
T0*
_output_shapes
:	2
activation_23/Sigmoide
IdentityIdentityactivation_23/Sigmoid:y:0*
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
Ó
k
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10063

inputs
identityÙ
activation_16/PartitionedCallPartitionedCallinputs*
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
G__inference_activation_16_layer_call_and_return_conditional_losses_99412
activation_16/PartitionedCallù
activation_17/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
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
G__inference_activation_17_layer_call_and_return_conditional_losses_99482
activation_17/PartitionedCallù
activation_18/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
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
G__inference_activation_18_layer_call_and_return_conditional_losses_99552
activation_18/PartitionedCallù
activation_19/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
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
G__inference_activation_19_layer_call_and_return_conditional_losses_99622
activation_19/PartitionedCallù
activation_20/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
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
G__inference_activation_20_layer_call_and_return_conditional_losses_99692
activation_20/PartitionedCallù
activation_21/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
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
G__inference_activation_21_layer_call_and_return_conditional_losses_99762
activation_21/PartitionedCallù
activation_22/PartitionedCallPartitionedCall&activation_21/PartitionedCall:output:0*
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
G__inference_activation_22_layer_call_and_return_conditional_losses_99832
activation_22/PartitionedCallù
activation_23/PartitionedCallPartitionedCall&activation_22/PartitionedCall:output:0*
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
G__inference_activation_23_layer_call_and_return_conditional_losses_99902
activation_23/PartitionedCallr
IdentityIdentity&activation_23/PartitionedCall:output:0*
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
Á
c
G__inference_activation_20_layer_call_and_return_conditional_losses_9969

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
Á
c
G__inference_activation_18_layer_call_and_return_conditional_losses_9955

inputs
identityO
SigmoidSigmoidinputs*
T0*
_output_shapes
:	2	
SigmoidW
IdentityIdentitySigmoid:y:0*
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
activation_23 
PartitionedCall:0	tensorflow/serving/predict:·»
¸)
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
^_default_save_signature"'
_tf_keras_networkë&{"name": "sigmoid_act_stacked8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "sigmoid_act_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_16", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_17", "inbound_nodes": [[["activation_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_18", "inbound_nodes": [[["activation_17", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_19", "inbound_nodes": [[["activation_18", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_20", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_21", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_22", "inbound_nodes": [[["activation_21", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_23", "inbound_nodes": [[["activation_22", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_23", 0, 0]]}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "sigmoid_act_stacked8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_16", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_17", "inbound_nodes": [[["activation_16", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_18", "inbound_nodes": [[["activation_17", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_19", "inbound_nodes": [[["activation_18", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_20", "inbound_nodes": [[["activation_19", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_21", "inbound_nodes": [[["activation_20", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_22", "inbound_nodes": [[["activation_21", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "activation_23", "inbound_nodes": [[["activation_22", 0, 0, {}]]], "shared_object_id": 8}], "input_layers": [["input_2", 0, 0]], "output_layers": [["activation_23", 0, 0]]}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}

trainable_variables
regularization_losses
	variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"
_tf_keras_layerô{"name": "activation_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1}
£
trainable_variables
regularization_losses
	variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"
_tf_keras_layerú{"name": "activation_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["activation_16", 0, 0, {}]]], "shared_object_id": 2}
£
trainable_variables
regularization_losses
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"
_tf_keras_layerú{"name": "activation_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["activation_17", 0, 0, {}]]], "shared_object_id": 3}
£
trainable_variables
regularization_losses
	variables
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"
_tf_keras_layerú{"name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["activation_18", 0, 0, {}]]], "shared_object_id": 4}
£
trainable_variables
 regularization_losses
!	variables
"	keras_api
*g&call_and_return_all_conditional_losses
h__call__"
_tf_keras_layerú{"name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["activation_19", 0, 0, {}]]], "shared_object_id": 5}
£
#trainable_variables
$regularization_losses
%	variables
&	keras_api
*i&call_and_return_all_conditional_losses
j__call__"
_tf_keras_layerú{"name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["activation_20", 0, 0, {}]]], "shared_object_id": 6}
£
'trainable_variables
(regularization_losses
)	variables
*	keras_api
*k&call_and_return_all_conditional_losses
l__call__"
_tf_keras_layerú{"name": "activation_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_22", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["activation_21", 0, 0, {}]]], "shared_object_id": 7}
£
+trainable_variables
,regularization_losses
-	variables
.	keras_api
*m&call_and_return_all_conditional_losses
n__call__"
_tf_keras_layerú{"name": "activation_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_23", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "inbound_nodes": [[["activation_22", 0, 0, {}]]], "shared_object_id": 8}
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
2
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10114
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10126
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10083
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10095À
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
2
3__inference_sigmoid_act_stacked8_layer_call_fn_9996
4__inference_sigmoid_act_stacked8_layer_call_fn_10131
4__inference_sigmoid_act_stacked8_layer_call_fn_10136
4__inference_sigmoid_act_stacked8_layer_call_fn_10071À
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
__inference__wrapped_model_9929®
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
ò2ï
H__inference_activation_16_layer_call_and_return_conditional_losses_10141¢
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
×2Ô
-__inference_activation_16_layer_call_fn_10146¢
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
ò2ï
H__inference_activation_17_layer_call_and_return_conditional_losses_10151¢
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
×2Ô
-__inference_activation_17_layer_call_fn_10156¢
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
ò2ï
H__inference_activation_18_layer_call_and_return_conditional_losses_10161¢
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
×2Ô
-__inference_activation_18_layer_call_fn_10166¢
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
ò2ï
H__inference_activation_19_layer_call_and_return_conditional_losses_10171¢
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
×2Ô
-__inference_activation_19_layer_call_fn_10176¢
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
ò2ï
H__inference_activation_20_layer_call_and_return_conditional_losses_10181¢
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
×2Ô
-__inference_activation_20_layer_call_fn_10186¢
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
ò2ï
H__inference_activation_21_layer_call_and_return_conditional_losses_10191¢
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
×2Ô
-__inference_activation_21_layer_call_fn_10196¢
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
ò2ï
H__inference_activation_22_layer_call_and_return_conditional_losses_10201¢
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
×2Ô
-__inference_activation_22_layer_call_fn_10206¢
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
ò2ï
H__inference_activation_23_layer_call_and_return_conditional_losses_10211¢
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
×2Ô
-__inference_activation_23_layer_call_fn_10216¢
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
#__inference_signature_wrapper_10102input_2"
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
__inference__wrapped_model_9929a(¢%
¢

input_2	
ª "5ª2
0
activation_23
activation_23	
H__inference_activation_16_layer_call_and_return_conditional_losses_10141H'¢$
¢

inputs	
ª "¢

0	
 l
-__inference_activation_16_layer_call_fn_10146;'¢$
¢

inputs	
ª "	
H__inference_activation_17_layer_call_and_return_conditional_losses_10151H'¢$
¢

inputs	
ª "¢

0	
 l
-__inference_activation_17_layer_call_fn_10156;'¢$
¢

inputs	
ª "	
H__inference_activation_18_layer_call_and_return_conditional_losses_10161H'¢$
¢

inputs	
ª "¢

0	
 l
-__inference_activation_18_layer_call_fn_10166;'¢$
¢

inputs	
ª "	
H__inference_activation_19_layer_call_and_return_conditional_losses_10171H'¢$
¢

inputs	
ª "¢

0	
 l
-__inference_activation_19_layer_call_fn_10176;'¢$
¢

inputs	
ª "	
H__inference_activation_20_layer_call_and_return_conditional_losses_10181H'¢$
¢

inputs	
ª "¢

0	
 l
-__inference_activation_20_layer_call_fn_10186;'¢$
¢

inputs	
ª "	
H__inference_activation_21_layer_call_and_return_conditional_losses_10191H'¢$
¢

inputs	
ª "¢

0	
 l
-__inference_activation_21_layer_call_fn_10196;'¢$
¢

inputs	
ª "	
H__inference_activation_22_layer_call_and_return_conditional_losses_10201H'¢$
¢

inputs	
ª "¢

0	
 l
-__inference_activation_22_layer_call_fn_10206;'¢$
¢

inputs	
ª "	
H__inference_activation_23_layer_call_and_return_conditional_losses_10211H'¢$
¢

inputs	
ª "¢

0	
 l
-__inference_activation_23_layer_call_fn_10216;'¢$
¢

inputs	
ª "	¤
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10083Q0¢-
&¢#

input_2	
p 

 
ª "¢

0	
 ¤
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10095Q0¢-
&¢#

input_2	
p

 
ª "¢

0	
 £
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10114P/¢,
%¢"

inputs	
p 

 
ª "¢

0	
 £
O__inference_sigmoid_act_stacked8_layer_call_and_return_conditional_losses_10126P/¢,
%¢"

inputs	
p

 
ª "¢

0	
 |
4__inference_sigmoid_act_stacked8_layer_call_fn_10071D0¢-
&¢#

input_2	
p

 
ª "	{
4__inference_sigmoid_act_stacked8_layer_call_fn_10131C/¢,
%¢"

inputs	
p 

 
ª "	{
4__inference_sigmoid_act_stacked8_layer_call_fn_10136C/¢,
%¢"

inputs	
p

 
ª "	{
3__inference_sigmoid_act_stacked8_layer_call_fn_9996D0¢-
&¢#

input_2	
p 

 
ª "	
#__inference_signature_wrapper_10102l3¢0
¢ 
)ª&
$
input_2
input_2	"5ª2
0
activation_23
activation_23	