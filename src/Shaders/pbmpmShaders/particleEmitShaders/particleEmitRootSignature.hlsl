#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=34, b0),"     /* 32 constants: w/ one int2, see PBMPM Constants */ \
"DescriptorTable(SRV(t0, numDescriptors=1))," /* For Sim Shapes */ \
"DescriptorTable(UAV(u0, numDescriptors=3)),"    /* Table for particleBuffer, freeIndicesBuffer, particleCountBuffer */ \
"DescriptorTable(SRV(t1, numDescriptors=1)), " /* Table for curr grid */ \
"DescriptorTable(UAV(u3, numDescriptors=2))" /* Table for g_positions, g_materials */