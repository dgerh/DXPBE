#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=24, b0),"     /* 17 constants: w/ one int2, see PBMPM Constants */ \
"DescriptorTable(UAV(u0, numDescriptors=2)),"    /* Table for particleBuffer, freeIndicesBuffer */ \
"DescriptorTable(SRV(t0, numDescriptors=2))," /* Table for BukkitParticleData & ThreadData */ \
"DescriptorTable(SRV(t2, numDescriptors=1))," /* Table for curr grid */ \
"DescriptorTable(UAV(u2, numDescriptors=1))," /* Table for next grid */ \
"DescriptorTable(UAV(u3, numDescriptors=1))," /* Table for nextnext grid */ \
"DescriptorTable(UAV(u4, numDescriptors=1))" /* Table for tempTileData */

