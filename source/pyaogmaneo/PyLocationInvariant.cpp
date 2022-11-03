// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyLocationInvariant.h"

using namespace pyaon;

void LocationInvariant::initCheck() const {
    if (!initialized) {
        std::cerr << "Attempted to use the LocationInvariant uninitialized!" << std::endl;
        abort();
    }
}

void LocationInvariant::initRandom(
    const std::tuple<int, int, int> &hiddenSize,
    const std::tuple<int, int> &intermSize,
    int sensorSizeZ,
    int whereSizeZ,
    int radius
) {
    bool allInRange = true;

    if (std::get<0>(hiddenSize) < 1) {
        std::cerr << "Error: hiddenSize[0] < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<1>(hiddenSize) < 1) {
        std::cerr << "Error: hiddenSize[1] < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<2>(hiddenSize) < 1) {
        std::cerr << "Error: hiddenSize[2] < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<0>(intermSize) < 1) {
        std::cerr << "Error: intermSize[0] < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<1>(intermSize) < 1) {
        std::cerr << "Error: intermSize[1] < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (sensorSizeZ < 1) {
        std::cerr << "Error: sensorSizeZ < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (whereSizeZ < 1) {
        std::cerr << "Error: whereSizeZ < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (radius < 0) {
        std::cerr << "Error: radius < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (!allInRange) {
        std::cerr << " - LocationInvariant: Some parameters out of range!" << std::endl;
        abort();
    }

    li.initRandom(aon::Int3(std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize)), aon::Int2(std::get<0>(intermSize), std::get<1>(intermSize)), sensorSizeZ, whereSizeZ, radius);

    initialized = true;
}

void LocationInvariant::initFromFile(
    const std::string &name
) {
    FileReader reader;
    reader.ins.open(name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != locationInvariantMagic) {
        std::cerr << "Attempted to initialize LocationInvariant from incompatible file - " << name << std::endl;
        abort();
    }

    li.read(reader);

    initialized = true;
}

void LocationInvariant::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != locationInvariantMagic) {
        std::cerr << "Attempted to initialize LocationInvariant from incompatible buffer!" << std::endl;
        abort();
    }

    li.read(reader);

    initialized = true;
}

void LocationInvariant::saveToFile(
    const std::string &name
) {
    initCheck();

    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&locationInvariantMagic, sizeof(int));

    li.write(writer);
}

std::vector<unsigned char> LocationInvariant::serializeToBuffer() {
    initCheck();

    BufferWriter writer(li.size() + sizeof(int));

    writer.write(&locationInvariantMagic, sizeof(int));

    li.write(writer);

    return writer.buffer;
}

void LocationInvariant::step(
    const std::vector<int> &sensorCIs,
    const std::vector<int> &whereCIs,
    bool learnEnabled
) {
    initCheck();

    int numIntermColumns = li.getIntermSize().x * li.getIntermSize().y;

    if (sensorCIs.size() != numIntermColumns) {
        std::cerr << "Incorrect number of sensorCIs given to LocationInvariant! Expected " << numIntermColumns << ", got " << sensorCIs.size() << std::endl;
        abort();
    }

    if (whereCIs.size() != numIntermColumns) {
        std::cerr << "Incorrect number of whereCIs given to LocationInvariant! Expected " << numIntermColumns << ", got " << whereCIs.size() << std::endl;
        abort();
    }

    aon::IntBuffer cSensorCIs(numIntermColumns);
    aon::IntBuffer cWhereCIs(numIntermColumns);

    for (int j = 0; j < numIntermColumns; j++) {
        if (sensorCIs[j] < 0 || sensorCIs[j] >= li.getSensorSizeZ()) {
            std::cerr << "Sensor CSDR has an out-of-bounds column index (" << sensorCIs[j] << ") at column index " << j << ". It must be in the range [0, " << (li.getSensorSizeZ() - 1) << "]" << std::endl;
            abort();
        }

        cSensorCIs[j] = sensorCIs[j];

        if (whereCIs[j] < 0 || whereCIs[j] >= li.getWhereSizeZ()) {
            std::cerr << "Sensor CSDR has an out-of-bounds column index (" << whereCIs[j] << ") at column index " << j << ". It must be in the range [0, " << (li.getWhereSizeZ() - 1) << "]" << std::endl;
            abort();
        }

        cWhereCIs[j] = whereCIs[j];
    }

    li.step(&cSensorCIs, &cWhereCIs, learnEnabled);
}
