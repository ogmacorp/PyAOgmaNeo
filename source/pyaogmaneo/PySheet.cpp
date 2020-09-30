// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PySheet.h"

using namespace pyaon;

PySheet::PySheet(
    const std::vector<PySheetInputDesc> &inputDescs,
    int recurrentRadius,
    const std::vector<PySheetOutputDesc> &outputDescs,
    const PyInt3 &actorSize
) {
    aon::Array<aon::Sheet::InputDesc> cInputDescs(inputDescs.size());

    for (int i = 0; i < inputDescs.size(); i++) {
        cInputDescs[i].size = aon::Int3(inputDescs[i].size.x, inputDescs[i].size.y, inputDescs[i].size.z);
        cInputDescs[i].radius = inputDescs[i].radius;
        cInputDescs[i].recurrent = inputDescs[i].recurrent;
    }
    
    aon::Array<aon::Sheet::OutputDesc> cOutputDescs(outputDescs.size());

    for (int i = 0; i < outputDescs.size(); i++) {
        cOutputDescs[i].size = aon::Int3(outputDescs[i].size.x, outputDescs[i].size.y, outputDescs[i].size.z);
        cOutputDescs[i].radius = outputDescs[i].radius;
    }

    s.initRandom(cInputDescs, recurrentRadius, cOutputDescs, aon::Int3(actorSize.x, actorSize.y, actorSize.z));
}

PySheet::PySheet(
    const std::string &name
) {
    PyStreamReader reader;
    reader.ins.open(name, std::ios::binary);

    s.read(reader);
}

PySheet::PySheet(
    const std::vector<unsigned char> &buffer
) {
    PyBufferReader reader;
    reader.buffer = &buffer;

    s.read(reader);
}

void PySheet::save(
    const std::string &name
) {
    PyStreamWriter writer;
    writer.outs.open(name, std::ios::binary);

    s.write(writer);
}

std::vector<unsigned char> PySheet::save() {
    PyBufferWriter writer;

    s.write(writer);

    return writer.buffer;
}

std::vector<std::vector<int> > PySheet::step(
    const std::vector<std::vector<int> > &inputCs,
    const std::vector<std::vector<int> > &targetCs,
    int subSteps,
    bool learnEnabled
) {
    aon::Array<aon::IntBuffer> cInputCsBacking(inputCs.size());
    aon::Array<const aon::IntBuffer*> cInputCs(inputCs.size());

    aon::Array<aon::IntBuffer> cTargetCsBacking(targetCs.size());
    aon::Array<const aon::IntBuffer*> cTargetCs(targetCs.size());

    for (int i = 0; i < inputCs.size(); i++) {
        cInputCsBacking[i].resize(inputCs[i].size());

        for (int j = 0; j < inputCs[i].size(); j++)
            cInputCsBacking[i][j] = inputCs[i][j];

        cInputCs[i] = &cInputCsBacking[i];
    }
    
    for (int i = 0; i < targetCs.size(); i++) {
        cTargetCsBacking[i].resize(targetCs[i].size());

        for (int j = 0; j < targetCs[i].size(); j++)
            cTargetCsBacking[i][j] = targetCs[i][j];

        cTargetCs[i] = &cTargetCsBacking[i];
    }

    aon::Array<aon::IntBuffer> cIntermediates(subSteps);

    s.step(cInputCs, cTargetCs, cIntermediates, learnEnabled);

    std::vector<std::vector<int>> intermediates(cIntermediates.size());

    for (int i = 0; i < cIntermediates.size(); i++) {
        intermediates[i].resize(cIntermediates[i].size());

        for (int j = 0; j < cIntermediates[i].size(); j++)
            intermediates[i][j] = cIntermediates[i][j];
    }

    return intermediates;
}
