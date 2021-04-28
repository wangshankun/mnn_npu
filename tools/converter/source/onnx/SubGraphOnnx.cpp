//
//  SubGraph.cpp
//  MNNConverter
//
//  Created by MNN on 2019/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#include "MNN/plugin/PluginShapeInference.hpp"
#include "MNN_generated.h"
DECLARE_OP_CONVERTER(SubGraphOnnx);

MNN::OpType SubGraphOnnx::opType() {
    return MNN::OpType_Plugin;
}

MNN::OpParameter SubGraphOnnx::type() {
    return MNN::OpParameter_Plugin;
}

void SubGraphOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      std::vector<const onnx::TensorProto *> initializers) {
    MNN::PluginT* plugin_param = new MNN::PluginT;
    plugin_param->type    = "SubGraph";
    plugin_param->attr.resize(5);
    //加载的file 默认就是子图name，执行时候放到mnn默认路径下
    plugin_param->attr[0].reset(new MNN::AttributeT);
    plugin_param->attr[0]->key = "conf_file";
    plugin_param->attr[0]->s   = onnxNode->name();
    //SubGraph子图类型插件，数输出不固定，没有计算公式，只能从onnx本身属性中获取
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "inputs_info") {
            plugin_param->attr[1].reset(new MNN::AttributeT);
            plugin_param->attr[1]->key = "inputs_info";
            plugin_param->attr[1]->s   = attributeProto.s();
        }
        if (attributeName == "outputs_info") {
            plugin_param->attr[2].reset(new MNN::AttributeT);
            plugin_param->attr[2]->key = "outputs_info";
            plugin_param->attr[2]->s   = attributeProto.s();
        }
    }

    //记录input output 顺序与 name对应 信息
    std::ostringstream istr_maps;
    for(int i = 0; i < onnxNode->input_size(); i++)
    {
        istr_maps << onnxNode->input(i) << ":" << i << ";";
    }
    plugin_param->attr[3].reset(new MNN::AttributeT);
    plugin_param->attr[3]->key = "input_index_map";
    plugin_param->attr[3]->s   = istr_maps.str();

    std::ostringstream ostr_maps;
    for(int i = 0; i < onnxNode->output_size(); i++)
    {
        ostr_maps << onnxNode->output(i) << ":" << i << ";";
    }
    plugin_param->attr[4].reset(new MNN::AttributeT);
    plugin_param->attr[4]->key = "output_index_map";
    plugin_param->attr[4]->s   = ostr_maps.str();

    dstOp->type = MNN::OpType_Plugin;
    dstOp->main.type  = MNN::OpParameter_Plugin;
    dstOp->main.value = plugin_param;
    return;
}

REGISTER_CONVERTER(SubGraphOnnx, SubGraph);
