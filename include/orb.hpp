#ifndef ORB_H
#define ORB_H

struct Keypoint { int x, y; };

struct ORBDescriptor {
    uint8_t data[32];
};

#endif // ORB_H