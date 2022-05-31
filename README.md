# Heterogenous-Inferencing
Project is Part of the ["Hardware Accelerators for AI - Hands On"](https://elearning.ovgu.de/course/view.php?id=11672) Course at the [Otto-von-Guericke University](https://www.ovgu.de/en/)

We try to implement an efficient way to evaluate heterogenous inferencing infrastructures for convolutional neural networks. Our heterogenous inferencing infrastructure will consist of an [Intel® Neural Compute Stick 2](https://ark.intel.com/content/www/de/de/ark/products/140109/intel-neural-compute-stick-2.html), an [USB Accelerator based on the Edge TPU from Coral](https://coral.ai/products/accelerator/) and Intel CPU and integrated Graphics Card.

We executed the following inference benchmarks to determine the performance of the different devices listed above: Energy Consumption for "Low-Power" USB AI-Accelerators, Runtime of asynchronous Batch-Inferences for Intel-based Hardware, Runtime of a Single Inference and synchronous Batch-Inferences for the entire Hardware

Textual Statistics of the measurements can be found in the [logs folder](logs).<br>
Estimations derived from the measurement results can be found in the [res folder](res).

Violinplots for all measurements can be found in the [violinplots folder](violinplots). They were created on a per model basis.<br>
Plots for aggregated statistic results can be found in the [plots folder](plots). They are aggregated in the three following categories: Single-Operations, Dense and Convolutional Layer.
