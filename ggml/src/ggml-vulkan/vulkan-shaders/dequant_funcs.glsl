#if !defined(DATA_A_F32) && !defined(DATA_A_F16)
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#endif

#include "types.glsl"

const float PI_COS[64] = float[](
    0.7386546135f,0.8607548475f,-0.7411674857f,0.9674890637f,-0.7723053098f,-0.8056974411f,-0.0412844308f,0.2707833052f,
    0.9315500855f,0.6698185802f,0.9167487621f,-0.8320636749f,0.6818146110f,-0.9108457565f,-0.0559285842f,-0.9032276273f,
    0.7519487143f,-0.8941103816f,-0.1039871648f,-0.6961420774f,-0.1230370328f,-0.9328963161f,-0.2905603051f,0.4910068214f,
    0.7889407277f,-0.1221836656f,-0.6316579580f,0.3128163815f,-0.9563610554f,0.9992509484f,0.9540294409f,0.8902468085f,
    0.7543080449f,-0.8664138913f,-0.5232898593f,0.3621287644f,-0.8825117350f,0.8234673142f,-0.9416025877f,-0.5480425358f,
    -0.6644080281f,-0.6585279703f,-0.2460795939f,0.9438471198f,0.2427810431f,-0.1960992366f,0.2403578013f,-0.8461306095f,
    0.0246123374f,0.3372744620f,0.9994974732f,-0.3494733870f,0.7438930869f,0.8452339768f,-0.6177822948f,-0.2662552595f,
    -0.5457068086f,-0.9985070229f,0.7757105827f,0.6141811609f,-0.9805000424f,0.5425475240f,-0.5663578510f,-0.4696439803f
);
const float PI_SIN[64] = float[](
    -0.6740840673f,-0.5090196729f,0.6713201404f,-0.2529129684f,0.6352515221f,-0.5923272967f,0.9991474152f,-0.9626403451f,
    -0.3636130989f,0.7425247431f,-0.3994642496f,-0.5546801090f,-0.7315250039f,-0.4127469361f,-0.9984347820f,0.4291617870f,
    -0.6592215896f,-0.4478466809f,0.9945786595f,-0.7179040313f,0.9924020767f,0.3601450622f,0.9568566680f,-0.8711557388f,
    0.6144692898f,0.9925075173f,0.7752471566f,0.9498136044f,-0.2921875417f,0.0386975110f,-0.2997128963f,0.4554784000f,
    -0.6565206647f,-0.4993265271f,0.8521547318f,-0.9321280718f,-0.4702904224f,-0.5673637390f,-0.3367263079f,0.8364504576f,
    -0.7473700047f,0.7525562644f,-0.9692496061f,-0.3303825557f,-0.9700810909f,0.9805840850f,-0.9706843495f,-0.5329755545f,
    -0.9996970892f,0.9414063692f,0.0316982083f,0.9369462729f,0.6682986617f,-0.5343964100f,-0.7863491774f,-0.9639025331f,
    -0.8379761577f,0.0546237342f,-0.6310887933f,0.7891650796f,-0.1965190321f,0.8400250673f,-0.8241594434f,0.8828558922f
);

const float C3B[8] = float[](
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
);

const float COS_C3B[512] = float[](
    -0.1408503550f, -0.0870371504f, -0.0485421652f, -0.0158515280f, 0.0158515280f, 0.0485421652f, 0.0870371504f, 0.1408503550f,
    -0.1641330381f, -0.1014244652f, -0.0565662263f, -0.0184717990f, 0.0184717990f, 0.0565662263f, 0.1014244652f, 0.1641330381f,
    0.1413295220f, 0.0873332472f, 0.0487073037f, 0.0159054542f, -0.0159054542f, -0.0487073037f, -0.0873332472f, -0.1413295220f,
    -0.1844856521f, -0.1140011714f, -0.0635804788f, -0.0207623153f, 0.0207623153f, 0.0635804788f, 0.1140011714f, 0.1844856521f,
    0.1472670380f, 0.0910022793f, 0.0507535880f, 0.0165736719f, -0.0165736719f, -0.0507535880f, -0.0910022793f, -0.1472670380f,
    0.1536344166f, 0.0949369409f, 0.0529480187f, 0.0172902671f, -0.0172902671f, -0.0529480187f, -0.0949369409f, -0.1536344166f,
    0.0078723217f, 0.0048646271f, 0.0027130889f, 0.0008859639f, -0.0008859639f, -0.0027130889f, -0.0048646271f, -0.0078723217f,
    -0.0516343146f, -0.0319069384f, -0.0177950665f, -0.0058110097f, 0.0058110097f, 0.0177950665f, 0.0319069384f, 0.0516343146f,
    -0.1776326281f, -0.1097664097f, -0.0612186770f, -0.0199910648f, 0.0199910648f, 0.0612186770f, 0.1097664097f, 0.1776326281f,
    -0.1277243560f, -0.0789260629f, -0.0440184676f, -0.0143743067f, 0.0143743067f, 0.0440184676f, 0.0789260629f, 0.1277243560f,
    -0.1748102377f, -0.1080223401f, -0.0602459784f, -0.0196734284f, 0.0196734284f, 0.0602459784f, 0.1080223401f, 0.1748102377f,
    0.1586620618f, 0.0980437269f, 0.0546807285f, 0.0178560865f, -0.0178560865f, -0.0546807285f, -0.0980437269f, -0.1586620618f,
    -0.1300118191f, -0.0803395792f, -0.0448068108f, -0.0146317416f, 0.0146317416f, 0.0448068108f, 0.0803395792f, 0.1300118191f,
    0.1736846231f, 0.1073267772f, 0.0598580506f, 0.0195467499f, -0.0195467499f, -0.0598580506f, -0.1073267772f, -0.1736846231f,
    0.0106647421f, 0.0065901769f, 0.0036754588f, 0.0012002274f, -0.0012002274f, -0.0036754588f, -0.0065901769f, -0.0106647421f,
    0.1722319601f, 0.1064291178f, 0.0593574100f, 0.0193832649f, -0.0193832649f, -0.0593574100f, -0.1064291178f, -0.1722319601f,
    -0.1433853406f, -0.0886036209f, -0.0494158137f, -0.0161368194f, 0.0161368194f, 0.0494158137f, 0.0886036209f, 0.1433853406f,
    0.1704934381f, 0.1053548145f, 0.0587582519f, 0.0191876088f, -0.0191876088f, -0.0587582519f, -0.1053548145f, -0.1704934381f,
    0.0198287925f, 0.0122530156f, 0.0068337245f, 0.0022315646f, -0.0022315646f, -0.0068337245f, -0.0122530156f, -0.0198287925f,
    0.1327438520f, 0.0820278133f, 0.0457483689f, 0.0149392090f, -0.0149392090f, -0.0457483689f, -0.0820278133f, -0.1327438520f,
    0.0234613166f, 0.0144976996f, 0.0080856247f, 0.0026403747f, -0.0026403747f, -0.0080856247f, -0.0144976996f, -0.0234613166f,
    0.1778893340f, 0.1099250387f, 0.0613071472f, 0.0200199549f, -0.0200199549f, -0.0613071472f, -0.1099250387f, -0.1778893340f,
    0.0554054918f, 0.0342373019f, 0.0190947516f, 0.0062354241f, -0.0062354241f, -0.0190947516f, -0.0342373019f, -0.0554054918f,
    -0.0936276357f, -0.0578563158f, -0.0322674953f, -0.0105370064f, 0.0105370064f, 0.0322674953f, 0.0578563158f, 0.0936276357f,
    -0.1504391627f, -0.0929624638f, -0.0518468178f, -0.0169306680f, 0.0169306680f, 0.0518468178f, 0.0929624638f, 0.1504391627f,
    0.0232985923f, 0.0143971457f, 0.0080295440f, 0.0026220615f, -0.0026220615f, -0.0080295440f, -0.0143971457f, -0.0232985923f,
    0.1204476977f, 0.0744295205f, 0.0415106660f, 0.0135553798f, -0.0135553798f, -0.0415106660f, -0.0744295205f, -0.1204476977f,
    -0.0596493917f, -0.0368597799f, -0.0205573541f, -0.0067130395f, 0.0067130395f, 0.0205573541f, 0.0368597799f, 0.0596493917f,
    0.1823637078f, 0.1126899359f, 0.0628491795f, 0.0205235082f, -0.0205235082f, -0.0628491795f, -0.1126899359f, -0.1823637078f,
    -0.1905421671f, -0.1177437378f, -0.0656677746f, -0.0214439254f, 0.0214439254f, 0.0656677746f, 0.1177437378f, 0.1905421671f,
    -0.1819191039f, -0.1124151971f, -0.0626959528f, -0.0204734718f, 0.0204734718f, 0.0626959528f, 0.1124151971f, 0.1819191039f,
    -0.1697567127f, -0.1048995619f, -0.0585043495f, -0.0191046965f, 0.0191046965f, 0.0585043495f, 0.1048995619f, 0.1697567127f,
    -0.1438352295f, -0.0888816255f, -0.0495708618f, -0.0161874506f, 0.0161874506f, 0.0495708618f, 0.0888816255f, 0.1438352295f,
    0.1652121329f, 0.1020912816f, 0.0569381217f, 0.0185932421f, -0.0185932421f, -0.0569381217f, -0.1020912816f, -0.1652121329f,
    0.0997835268f, 0.0616602907f, 0.0343890397f, 0.0112298004f, -0.0112298004f, -0.0343890397f, -0.0616602907f, -0.0997835268f,
    -0.0690525234f, -0.0426703566f, -0.0237980160f, -0.0077712833f, 0.0077712833f, 0.0237980160f, 0.0426703566f, 0.0690525234f,
    0.1682817502f, 0.1039881228f, 0.0579960237f, 0.0189387018f, -0.0189387018f, -0.0579960237f, -0.1039881228f, -0.1682817502f,
    -0.1570228648f, -0.0970308006f, -0.0541158015f, -0.0176716086f, 0.0176716086f, 0.0541158015f, 0.0970308006f, 0.1570228648f,
    0.1795494894f, 0.1109509161f, 0.0618792973f, 0.0202067915f, -0.0202067915f, -0.0618792973f, -0.1109509161f, -0.1795494894f,
    0.1045034909f, 0.0645769481f, 0.0360157113f, 0.0117609928f, -0.0117609928f, -0.0360157113f, -0.0645769481f, -0.1045034909f,
    0.1266926448f, 0.0782885268f, 0.0436629024f, 0.0142581963f, -0.0142581963f, -0.0436629024f, -0.0782885268f, -0.1266926448f,
    0.1255714060f, 0.0775956678f, 0.0432764826f, 0.0141320102f, -0.0141320102f, -0.0432764826f, -0.0775956678f, -0.1255714060f,
    0.0469236874f, 0.0289960507f, 0.0161716127f, 0.0052808681f, -0.0052808681f, -0.0161716127f, -0.0289960507f, -0.0469236874f,
    -0.1799774880f, -0.1112153938f, -0.0620268012f, -0.0202549592f, 0.0202549592f, 0.0620268012f, 0.1112153938f, 0.1799774880f,
    -0.0462947032f, -0.0286073759f, -0.0159548418f, -0.0052100812f, 0.0052100812f, 0.0159548418f, 0.0286073759f, 0.0462947032f,
    0.0373931829f, 0.0231067652f, 0.0128870535f, 0.0042082896f, -0.0042082896f, -0.0128870535f, -0.0231067652f, -0.0373931829f,
    -0.0458326273f, -0.0283218404f, -0.0157955936f, -0.0051580784f, 0.0051580784f, 0.0157955936f, 0.0283218404f, 0.0458326273f,
    0.1613444153f, 0.0997012620f, 0.0556051653f, 0.0181579629f, -0.0181579629f, -0.0556051653f, -0.0997012620f, -0.1613444153f,
    -0.0046932036f, -0.0029001209f, -0.0016174490f, -0.0005281808f, 0.0005281808f, 0.0016174490f, 0.0029001209f, 0.0046932036f,
    -0.0643131808f, -0.0397417244f, -0.0221646658f, -0.0072379100f, 0.0072379100f, 0.0221646658f, 0.0397417244f, 0.0643131808f,
    -0.1905891757f, -0.1177727863f, -0.0656839754f, -0.0214492158f, 0.0214492158f, 0.0656839754f, 0.1177727863f, 0.1905891757f,
    0.0666393328f, 0.0411791481f, 0.0229663426f, 0.0074996989f, -0.0074996989f, -0.0229663426f, -0.0411791481f, -0.0666393328f,
    -0.1418492533f, -0.0876544102f, -0.0488864220f, -0.0159639456f, 0.0159639456f, 0.0488864220f, 0.0876544102f, 0.1418492533f,
    -0.1611734409f, -0.0995956100f, -0.0555462413f, -0.0181387211f, 0.0181387211f, 0.0555462413f, 0.0995956100f, 0.1611734409f,
    0.1178018169f, 0.0727945234f, 0.0405987991f, 0.0132576080f, -0.0132576080f, -0.0405987991f, -0.0727945234f, -0.1178018169f,
    0.0507708842f, 0.0313733897f, 0.0174974969f, 0.0057138379f, -0.0057138379f, -0.0174974969f, -0.0313733897f, -0.0507708842f,
    0.1040581028f, 0.0643017247f, 0.0358622143f, 0.0117108681f, -0.0117108681f, -0.0358622143f, -0.0643017247f, -0.1040581028f,
    0.1904003117f, 0.1176560795f, 0.0656188860f, 0.0214279607f, -0.0214279607f, -0.0656188860f, -0.1176560795f, -0.1904003117f,
    -0.1479163725f, -0.0914035294f, -0.0509773724f, -0.0166467491f, 0.0166467491f, 0.0509773724f, 0.0914035294f, 0.1479163725f,
    -0.1171151347f, -0.0723701946f, -0.0403621434f, -0.0131803277f, 0.0131803277f, 0.0403621434f, 0.0723701946f, 0.1171151347f,
    0.1869666506f, 0.1155342810f, 0.0644355213f, 0.0210415309f, -0.0210415309f, -0.0644355213f, -0.1155342810f, -0.1869666506f,
    -0.1034556746f, -0.0639294598f, -0.0356545956f, -0.0116430699f, 0.0116430699f, 0.0356545956f, 0.0639294598f, 0.1034556746f,
    0.1079959468f, 0.0667350783f, 0.0372193389f, 0.0121540395f, -0.0121540395f, -0.0372193389f, -0.0667350783f, -0.1079959468f,
    0.0895540624f, 0.0553390895f, 0.0308635935f, 0.0100785598f, -0.0100785598f, -0.0308635935f, -0.0553390895f, -0.0895540624f
);

const float SIN_C3B[512] = float[](
    0.1285377204f, 0.0794286738f, 0.0442987827f, 0.0144658441f, -0.0144658441f, -0.0442987827f, -0.0794286738f, -0.1285377204f,
    0.0970624163f, 0.0599788061f, 0.0334512458f, 0.0109235622f, -0.0109235622f, -0.0334512458f, -0.0599788061f, -0.0970624163f,
    -0.1280106810f, -0.0791029948f, -0.0441171457f, -0.0144065302f, 0.0144065302f, 0.0441171457f, 0.0791029948f, 0.1280106810f,
    0.0482267094f, 0.0298012409f, 0.0166206815f, 0.0054275123f, -0.0054275123f, -0.0166206815f, -0.0298012409f, -0.0482267094f,
    -0.1211329365f, -0.0748529574f, -0.0417468243f, -0.0136324977f, 0.0136324977f, 0.0417468243f, 0.0748529574f, 0.1211329365f,
    0.1129479306f, 0.0697951100f, 0.0389259730f, 0.0127113438f, -0.0127113438f, -0.0389259730f, -0.0697951100f, -0.1129479306f,
    -0.1905224249f, -0.1177315382f, -0.0656609707f, -0.0214417035f, 0.0214417035f, 0.0656609707f, 0.1177315382f, 0.1905224249f,
    0.1835610742f, 0.1134298371f, 0.0632618356f, 0.0206582618f, -0.0206582618f, -0.0632618356f, -0.1134298371f, -0.1835610742f,
    0.0693355638f, 0.0428452587f, 0.0238955620f, 0.0078031371f, -0.0078031371f, -0.0238955620f, -0.0428452587f, -0.0693355638f,
    -0.1415883306f, -0.0874931755f, -0.0487964985f, -0.0159345810f, 0.0159345810f, 0.0487964985f, 0.0874931755f, 0.1415883306f,
    0.0761718404f, 0.0470696715f, 0.0262515921f, 0.0085725028f, -0.0085725028f, -0.0262515921f, -0.0470696715f, -0.0761718404f,
    0.1057691766f, 0.0653590666f, 0.0364519127f, 0.0119034351f, -0.0119034351f, -0.0364519127f, -0.0653590666f, -0.1057691766f,
    0.1394908454f, 0.0861970543f, 0.0480736287f, 0.0156985266f, -0.0156985266f, -0.0480736287f, -0.0861970543f, -0.1394908454f,
    0.0787046495f, 0.0486347970f, 0.0271244904f, 0.0088575492f, -0.0088575492f, -0.0271244904f, -0.0486347970f, -0.0787046495f,
    0.1903865364f, 0.1176475672f, 0.0656141386f, 0.0214264104f, -0.0214264104f, -0.0656141386f, -0.1176475672f, -0.1903865364f,
    -0.0818347154f, -0.0505689917f, -0.0282032252f, -0.0092098119f, 0.0092098119f, 0.0282032252f, 0.0505689917f, 0.0818347154f,
    0.1257036688f, 0.0776773983f, 0.0433220652f, 0.0141468953f, -0.0141468953f, -0.0433220652f, -0.0776773983f, -0.1257036688f,
    0.0853976443f, 0.0527706701f, 0.0294311403f, 0.0096107898f, -0.0096107898f, -0.0294311403f, -0.0527706701f, -0.0853976443f,
    -0.1896512317f, -0.1171931926f, -0.0653607258f, -0.0213436580f, 0.0213436580f, 0.0653607258f, 0.1171931926f, 0.1896512317f,
    0.1368935302f, 0.0845920678f, 0.0471784992f, 0.0154062205f, -0.0154062205f, -0.0471784992f, -0.0845920678f, -0.1368935302f,
    -0.1892361900f, -0.1169367215f, -0.0652176873f, -0.0212969486f, 0.0212969486f, 0.0652176873f, 0.1169367215f, 0.1892361900f,
    -0.0686742612f, -0.0424366130f, -0.0236676531f, -0.0077287130f, 0.0077287130f, 0.0236676531f, 0.0424366130f, 0.0686742612f,
    -0.1824582137f, -0.1127483349f, -0.0628817497f, -0.0205341441f, 0.0205341441f, 0.0628817497f, 0.1127483349f, 0.1824582137f,
    0.1661163321f, 0.1026500230f, 0.0572497417f, 0.0186950022f, -0.0186950022f, -0.0572497417f, -0.1026500230f, -0.1661163321f,
    -0.1171700765f, -0.0724041454f, -0.0403810783f, -0.0131865110f, 0.0131865110f, 0.0403810783f, 0.0724041454f, 0.1171700765f,
    -0.1892562959f, -0.1169491458f, -0.0652246165f, -0.0212992113f, 0.0212992113f, 0.0652246165f, 0.1169491458f, 0.1892562959f,
    -0.1478280041f, -0.0913489230f, -0.0509469174f, -0.0166368040f, 0.0166368040f, 0.0509469174f, 0.0913489230f, 0.1478280041f,
    -0.1811152072f, -0.1119184366f, -0.0624189006f, -0.0203830000f, 0.0203830000f, 0.0624189006f, 0.1119184366f, 0.1811152072f,
    0.0557157814f, 0.0344290424f, 0.0192016887f, 0.0062703446f, -0.0062703446f, -0.0192016887f, -0.0344290424f, -0.0557157814f,
    -0.0073790349f, -0.0045598051f, -0.0025430843f, -0.0008304486f, 0.0008304486f, 0.0025430843f, 0.0045598051f, 0.0073790349f,
    0.0571507536f, 0.0353157700f, 0.0196962324f, 0.0064318388f, -0.0064318388f, -0.0196962324f, -0.0353157700f, -0.0571507536f,
    -0.0868528987f, -0.0536699308f, -0.0299326740f, -0.0097745665f, 0.0097745665f, 0.0299326740f, 0.0536699308f, 0.0868528987f,
    0.1251886429f, 0.0773591430f, 0.0431445685f, 0.0140889335f, -0.0140889335f, -0.0431445685f, -0.0773591430f, -0.1251886429f,
    0.0952140788f, 0.0588366433f, 0.0328142414f, 0.0107155473f, -0.0107155473f, -0.0328142414f, -0.0588366433f, -0.0952140788f,
    -0.1624931250f, -0.1004110964f, -0.0560010525f, -0.0182872405f, 0.0182872405f, 0.0560010525f, 0.1004110964f, 0.1624931250f,
    0.1777428414f, 0.1098345150f, 0.0612566605f, 0.0200034684f, -0.0200034684f, -0.0612566605f, -0.1098345150f, -0.1777428414f,
    0.0896773292f, 0.0554152611f, 0.0309060757f, 0.0100924325f, -0.0100924325f, -0.0309060757f, -0.0554152611f, -0.0896773292f,
    0.1081877546f, 0.0668536041f, 0.0372854428f, 0.0121756258f, -0.0121756258f, -0.0372854428f, -0.0668536041f, -0.1081877546f,
    0.0642086560f, 0.0396771343f, 0.0221286428f, 0.0072261466f, -0.0072261466f, -0.0221286428f, -0.0396771343f, -0.0642086560f,
    -0.1594985555f, -0.0985606303f, -0.0549690147f, -0.0179502268f, 0.0179502268f, 0.0549690147f, 0.0985606303f, 0.1594985555f,
    0.1425122493f, 0.0880641024f, 0.0491149146f, 0.0160385603f, -0.0160385603f, -0.0491149146f, -0.0880641024f, -0.1425122493f,
    -0.1435011913f, -0.0886752097f, -0.0494557400f, -0.0161498574f, 0.0161498574f, 0.0494557400f, 0.0886752097f, 0.1435011913f,
    0.1848213611f, 0.1142086196f, 0.0636961764f, 0.0208000965f, -0.0208000965f, -0.0636961764f, -0.1142086196f, -0.1848213611f,
    0.0629989976f, 0.0389296373f, 0.0217117504f, 0.0070900096f, -0.0070900096f, -0.0217117504f, -0.0389296373f, -0.0629989976f,
    0.1849799128f, 0.1143065951f, 0.0637508191f, 0.0208179402f, -0.0208179402f, -0.0637508191f, -0.1143065951f, -0.1849799128f,
    -0.1869826762f, -0.1155441839f, -0.0644410443f, -0.0210433345f, 0.0210433345f, 0.0644410443f, 0.1155441839f, 0.1869826762f,
    0.1850949452f, 0.1143776783f, 0.0637904634f, 0.0208308861f, -0.0208308861f, -0.0637904634f, -0.1143776783f, -0.1850949452f,
    0.1016304436f, 0.0628015755f, 0.0350255545f, 0.0114376554f, -0.0114376554f, -0.0350255545f, -0.0628015755f, -0.1016304436f,
    0.1906272395f, 0.1177963074f, 0.0656970936f, 0.0214534995f, -0.0214534995f, -0.0656970936f, -0.1177963074f, -0.1906272395f,
    -0.1795120735f, -0.1109277953f, -0.0618664024f, -0.0202025807f, 0.0202025807f, 0.0618664024f, 0.1109277953f, 0.1795120735f,
    -0.0060443728f, -0.0037350633f, -0.0020831112f, -0.0006802436f, 0.0006802436f, 0.0020831112f, 0.0037350633f, 0.0060443728f,
    -0.1786616000f, -0.1104022532f, -0.0615732982f, -0.0201068670f, 0.0201068670f, 0.0615732982f, 0.1104022532f, 0.1786616000f,
    -0.1274345303f, -0.0787469679f, -0.0439185832f, -0.0143416893f, 0.0143416893f, 0.0439185832f, 0.0787469679f, 0.1274345303f,
    0.1019013794f, 0.0629689978f, 0.0351189289f, 0.0114681470f, -0.0114681470f, -0.0351189289f, -0.0629689978f, -0.1019013794f,
    0.1499449929f, 0.0926570963f, 0.0516765089f, 0.0168750533f, -0.0168750533f, -0.0516765089f, -0.0926570963f, -0.1499449929f,
    0.1838017545f, 0.1135785633f, 0.0633447828f, 0.0206853484f, -0.0206853484f, -0.0633447828f, -0.1135785633f, -0.1838017545f,
    0.1597894836f, 0.0987404066f, 0.0550692792f, 0.0179829683f, -0.0179829683f, -0.0550692792f, -0.0987404066f, -0.1597894836f,
    -0.0104159268f, -0.0064364238f, -0.0035897079f, -0.0011722253f, 0.0011722253f, 0.0035897079f, 0.0064364238f, 0.0104159268f,
    0.1203391666f, 0.0743624547f, 0.0414732622f, 0.0135431655f, -0.0135431655f, -0.0414732622f, -0.0743624547f, -0.1203391666f,
    -0.1504819432f, -0.0929888997f, -0.0518615615f, -0.0169354826f, 0.0169354826f, 0.0518615615f, 0.0929888997f, 0.1504819432f,
    0.0374732316f, 0.0231562306f, 0.0129146412f, 0.0042172984f, -0.0042172984f, -0.0129146412f, -0.0231562306f, -0.0374732316f,
    -0.1601801800f, -0.0989818337f, -0.0552039273f, -0.0180269379f, 0.0180269379f, 0.0552039273f, 0.0989818337f, 0.1601801800f,
    0.1571548435f, 0.0971123555f, 0.0541612861f, 0.0176864617f, -0.0176864617f, -0.0541612861f, -0.0971123555f, -0.1571548435f,
    -0.1683473758f, -0.1040286755f, -0.0580186407f, -0.0189460874f,  0.0189460874f,  0.0580186407f,  0.1040286755f,  0.1683473758f
);

const float PI_QW[32] = float[]( 0.8350809813f,-0.1648498178f, 0.1283752173f, 0.2897698581f,-0.1820549369f, 0.9549587369f,-0.8741137385f, 0.8988990188f,-0.1312584430f,-0.3990598321f,-0.2694816887f,-0.1181898862f, 0.1363395452f, 0.2665117681f,-0.8263269663f,-0.1834189594f, 0.3098247349f, 0.2804697454f,-0.5655074716f,-0.1627507508f, 0.8684155941f, 0.2233296037f,-0.1291671842f, 0.6606932878f,-0.5694432259f,-0.2782760859f, 0.5113853812f,-0.5139024258f, 0.7489815354f,-0.3037399948f,-0.4143463373f,-0.3524050117f);
const float PI_QX[32] = float[]( 0.3547102809f,-0.5782636404f,-0.8299785256f, 0.5694668293f,-0.8199930191f, 0.1259543896f,-0.3090814352f,-0.2613596618f,-0.1660282463f,-0.5143862963f, 0.5898610353f,-0.8277072310f,-0.6826571226f,-0.1740629375f, 0.1416199356f, 0.4648889899f, 0.3485621810f, 0.8982698917f,-0.3015249372f, 0.4990116358f, 0.2398942262f,-0.7447698116f, 0.4783197045f, 0.0735855624f,-0.2975912094f,-0.0700704753f, 0.2975627482f,-0.2652103305f,-0.1539765000f, 0.0849994123f,-0.1069803685f,-0.5753474832f);
const float PI_QY[32] = float[]( 0.2416850179f,-0.4488199651f, 0.3478420675f, 0.5024775267f, 0.1696543097f, 0.1760476083f, 0.0254505407f, 0.2389279008f,-0.9429193735f, 0.3925755024f,-0.2757458389f,-0.1485267133f, 0.5530825853f,-0.8936085105f, 0.2953715622f,-0.5285226703f, 0.7939327955f, 0.0139789311f,-0.2555710375f, 0.4543992281f,-0.2698826790f,-0.4736968279f, 0.4361720681f,-0.3461222053f, 0.0792116225f, 0.8827795386f, 0.7416539788f,-0.3826399446f,-0.3534849286f,-0.8696597815f,-0.6908422709f, 0.2082736641f);
const float PI_QZ[32] = float[]( 0.3038694561f, 0.4734756052f,-0.3878843784f, 0.5831694603f,-0.5054479241f,-0.1731694490f,-0.3737666607f, 0.2328704894f, 0.2621760964f, 0.6239953637f,-0.7082104683f, 0.5308507681f,-0.4413037896f,-0.2802782655f,-0.4522367120f,-0.6698107123f,-0.3752456903f,-0.3359423280f, 0.7181019187f, 0.7106907368f, 0.3100073636f, 0.4016827941f, 0.7350437641f,-0.6607965231f, 0.7619289756f, 0.3648703992f,-0.3040413559f, 0.7213236690f, 0.5280022621f,-0.3742936850f,-0.5760775208f, 0.7015634775f);

#if defined(DATA_A_F32)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);
}
#endif

#if defined(DATA_A_F16)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);
}
#endif

#if defined(DATA_A_BF16)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(bf16_to_fp32(data_a[a_offset + ib]), bf16_to_fp32(data_a[a_offset + ib + 1]));
}
#endif

#if defined(DATA_A_Q4_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2(vui & 0xF, vui >> 4) - 8.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12) - 8.0f);
}
#endif

#if defined(DATA_A_Q4_1)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(vui & 0xF, vui >> 4);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12);
}
#endif

#if defined(DATA_A_Q5_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = uint(data_a[a_offset + ib].qh[1]) << 16 | data_a[a_offset + ib].qh[0];
    const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y) - 16.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = uint(data_a_packed16[a_offset + ib].qh[1]) << 16 | data_a_packed16[a_offset + ib].qh[0];
    const ivec2 qh0 = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const ivec2 qh1 = ivec2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) - 16.0f);
}
#endif

#if defined(DATA_A_Q5_1)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a[a_offset + ib].qh;
    const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a_packed16[a_offset + ib].qh;
    const ivec2 qh0 = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const ivec2 qh1 = ivec2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y);
}
#endif

#if defined(DATA_A_Q8_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(int(data_a[a_offset + ib].qs[iqs]), int(data_a[a_offset + ib].qs[iqs + 1]));
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const i8vec2 v0 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2])).xy; // vec4 used due to #12147
    const i8vec2 v1 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2 + 1])).xy;
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
#endif

#if defined(DATA_A_Q1_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 8u]) >> (iqs % 8u);
    return vec2(
        (bits & 1u) != 0u ? 1.0f : -1.0f,
        (bits & 2u) != 0u ? 1.0f : -1.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint bits = uint(data_a[a_offset + ib].qs[iqs / 8u]) >> (iqs % 8u);
    return vec4(
        (bits & 1u) != 0u ? 1.0f : -1.0f,
        (bits & 2u) != 0u ? 1.0f : -1.0f,
        (bits & 4u) != 0u ? 1.0f : -1.0f,
        (bits & 8u) != 0u ? 1.0f : -1.0f);
}
#endif

#if defined(DATA_A_TQ3_0)
const float tq3_centroids_df[8] = float[8](
    -2.1519454, -1.3439092, -0.7560052, -0.2450942,
     0.2450942,  0.7560052,  1.3439092,  2.1519454
);

uint tq3_extract_index(uint ib, uint val_pos, uint a_offset) {
    uint group = val_pos / 8u;
    uint within = val_pos % 8u;
    uint base = group * 3u;
    uint b0 = uint(data_a[a_offset + ib].qs[base + 0u]);
    uint b1 = uint(data_a[a_offset + ib].qs[base + 1u]);
    uint b2 = uint(data_a[a_offset + ib].qs[base + 2u]);

    uint idx;
    switch (within) {
        case 0u: idx =  b0       & 7u; break;
        case 1u: idx = (b0 >> 3) & 7u; break;
        case 2u: idx = ((b0 >> 6) | (b1 << 2)) & 7u; break;
        case 3u: idx = (b1 >> 1) & 7u; break;
        case 4u: idx = (b1 >> 4) & 7u; break;
        case 5u: idx = ((b1 >> 7) | (b2 << 1)) & 7u; break;
        case 6u: idx = (b2 >> 2) & 7u; break;
        default: idx = (b2 >> 5) & 7u; break;
    }
    return idx;
}

vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(
        tq3_centroids_df[tq3_extract_index(ib, iqs, a_offset)],
        tq3_centroids_df[tq3_extract_index(ib, iqs + 1u, a_offset)]
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(
        tq3_centroids_df[tq3_extract_index(ib, iqs, a_offset)],
        tq3_centroids_df[tq3_extract_index(ib, iqs + 1u, a_offset)],
        tq3_centroids_df[tq3_extract_index(ib, iqs + 2u, a_offset)],
        tq3_centroids_df[tq3_extract_index(ib, iqs + 3u, a_offset)]
    );
}
#endif

#if defined(DATA_A_IQ1_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;
    const int i8 = int(iqs % 8);
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const float dl = float(2 * bitfieldExtract(qh, 12, 3) + 1);
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint idxhi = bitfieldExtract(qh, 3 * int(ib8 & 3), 3);
    const int16_t grid = int16_t(iq1s_grid[qs | (idxhi << 8)]);
    // Signed bitfield extract.
    const ivec2 gvec = ivec2(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2)
    );
    return dl * (vec2(gvec) + delta);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;
    const int i8 = int(iqs % 8);
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const float dl = 2 * bitfieldExtract(qh, 12, 3) + 1;
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)]);
    // Signed bitfield extract.
    const ivec4 gvec = ivec4(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2),
      bitfieldExtract(grid, 2 * (i8 + 2), 2),
      bitfieldExtract(grid, 2 * (i8 + 3), 2)
    );
    return dl * (vec4(gvec) + delta);
}
#endif

#if defined(DATA_A_IQ1_M)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib8 = iqs / 8;
    const uint ib16 = iqs / 16;
    const int i8 = int(iqs % 8);
    const uint sc = data_a[a_offset + ib].scales[iqs / 64];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);
    // Signed bitfield extract.
    const ivec2 gvec = ivec2(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2)
    );
    return dl * (vec2(gvec) + delta);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib8 = iqs / 8;
    const uint ib16 = iqs / 16;
    const int i8 = int(iqs % 8);
    const uint sc = data_a[a_offset + ib].scales[iqs / 64];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);
    // Signed bitfield extract.
    const ivec4 gvec = ivec4(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2),
      bitfieldExtract(grid, 2 * (i8 + 2), 2),
      bitfieldExtract(grid, 2 * (i8 + 3), 2)
    );
    return dl * (vec4(gvec) + delta);
}
#endif

#if defined(DATA_A_IQ2_XXS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = (iqs / 8) % 4;
    const uint qs = data_a[a_offset + ib].qs[8 * ib32 + ib8];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[4 * ib32 + 2],
        data_a_packed16[a_offset + ib].qs[4 * ib32 + 3]));
    const float db = 0.25 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xxs_grid[qs][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = (iqs / 8) % 4;
    const uint qs = data_a[a_offset + ib].qs[8 * ib32 + ib8];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[4 * ib32 + 2],
        data_a_packed16[a_offset + ib].qs[4 * ib32 + 3]));
    const float db = 0.25 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xxs_grid[qs][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ2_XS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint scale = (data_a[a_offset + ib].scales[iqs / 32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[iqs / 8];
    const float db = 0.25 * (0.5 + scale);
    const uint sign7 = qs >> 9;
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xs_grid[qs & 511][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint scale = (data_a[a_offset + ib].scales[iqs / 32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[iqs / 8];
    const float db = 0.25 * (0.5 + scale);
    const uint sign7 = qs >> 9;
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xs_grid[qs & 511][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ2_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;

    const uint scale = (data_a[a_offset + ib].scales[ib32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qhshift = 2 * (ib8 % 4);
    const uint sign = data_a[a_offset + ib].qs[QUANT_K / 8 + ib8] >> (iqs % 8);

    const float db = 0.25 * (0.5 + scale);
    const u8vec4 grid = unpack8(iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(iqs % 8) / 4]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid[iqs % 4] * (sign0 ? -1.0 : 1.0),
        grid[(iqs % 4) + 1] * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;

    const uint scale = (data_a[a_offset + ib].scales[ib32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qhshift = 2 * (ib8 % 4);
    const uint sign = data_a[a_offset + ib].qs[QUANT_K / 8 + ib8] >> (iqs % 8);

    const float db = 0.25 * (0.5 + scale);
    const u8vec4 grid = unpack8(iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(iqs % 8) / 4]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ3_XXS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint is = QUANT_K / 4 + 4 * ib32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[is / 2],
        data_a_packed16[a_offset + ib].qs[is / 2 + 1]));
    const float db = 0.5 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * (int(ib4 / 2) % 4), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq3xxs_grid[qs] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint is = QUANT_K / 4 + 4 * ib32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[is / 2],
        data_a_packed16[a_offset + ib].qs[is / 2 + 1]));
    const float db = 0.5 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * (int(ib4 / 2) % 4), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq3xxs_grid[qs]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ3_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint qs = data_a[a_offset + ib].qs[iqs / 4];
    const uint qh = data_a[a_offset + ib].qh[iqs / 32];
    const uint sign = data_a[a_offset + ib].signs[iqs / 8] >> (iqs % 8);
    const uint scale = data_a[a_offset + ib].scales[iqs / 64];
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    const float db = 1 + 2 * ((scale >> (4 * ((iqs / 32) & 1))) & 0xf);
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - ((iqs / 4) % 8))) & 256)] >> (8 * (iqs % 4));
    return db * vec2(
        int(grid & 0xFF) * (sign0 ? -1.0 : 1.0),
        int((grid >> 8) & 0xFF) * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint sign = data_a[a_offset + ib].signs[iqs / 8] >> (iqs % 8);
    const uint scale = data_a[a_offset + ib].scales[ib32 / 2];
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    const float db = 1 + 2 * ((scale >> (4 * (ib32 & 1))) & 0xf);
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - ib4 % 8)) & 256)] >> (8 * (iqs % 4));
    return db * vec4(
        int(grid & 0xFF) * (sign0 ? -1.0 : 1.0),
        int((grid >> 8) & 0xFF) * (sign1 ? -1.0 : 1.0),
        int((grid >> 16) & 0xFF) * (sign2 ? -1.0 : 1.0),
        int((grid >> 24) & 0xFF) * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ4_XS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint iq = 16 * ib32 + (iqs % 16);

    const uint sl = (data_a[a_offset + ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3;
    const uint qshift = (iqs & 16) >> 2;
    u8vec2 qs = u8vec2(data_a[a_offset + ib].qs[iq], data_a[a_offset + ib].qs[iq + 1]);
    qs = (qs >> qshift) & uint8_t(0xF);

    const float dl = float(int(sl | (sh << 4)) - 32);
    return dl * vec2(kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint iq = 16 * ib32 + (iqs % 16);

    const uint sl = (data_a[a_offset + ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3;
    const uint qshift = (iqs & 16) >> 2;
    const u8vec4 qs = unpack8((data_a_packed32[a_offset + ib].qs[iq/4] >> qshift) & 0x0F0F0F0F);

    const float dl = float(int(sl | (sh << 4)) - 32);
    return dl * vec4(
        kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y],
        kvalues_iq4nl[qs.z], kvalues_iq4nl[qs.w]);
}
#endif

#if defined(DATA_A_IQ4_NL)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(kvalues_iq4nl[vui & 0xF], kvalues_iq4nl[vui >> 4]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4(kvalues_iq4nl[vui & 0xF], kvalues_iq4nl[(vui >> 4) & 0xF], kvalues_iq4nl[(vui >> 8) & 0xF], kvalues_iq4nl[vui >> 12]);
}
#endif

#if defined(DATA_A_MXFP4)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(kvalues_mxfp4[vui & 0xF], kvalues_mxfp4[vui >> 4]) * 0.5;
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    vec2 v0 = dequantize(ib, iqs, a_offset);
    vec2 v1 = dequantize(ib, iqs + 1, a_offset);
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
#endif

#if defined(DATA_A_NVFP4)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint sub = iqs >> 4;
    const float d = ue4m3_to_fp32(data_a[a_offset + ib].d[sub]);
    const uint j = iqs & 7;
    const uint shift = (iqs & 8) >> 1; // 0 or 4
    const uint vui0 = uint(data_a[a_offset + ib].qs[sub * 8u + j]);
    const uint vui1 = uint(data_a[a_offset + ib].qs[sub * 8u + j + 1]);
    const uint qs0 = (vui0 >> shift) & 0xF;
    const uint qs1 = (vui1 >> shift) & 0xF;
    return vec2(float(kvalues_mxfp4[qs0]), float(kvalues_mxfp4[qs1])) * d * 0.5;
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const vec2 v0 = dequantize(ib, iqs, a_offset);
    const vec2 v1 = dequantize(ib, iqs + 2u, a_offset);
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
#endif

#if defined(DATA_A_F32) || defined(DATA_A_F16) || defined(DATA_A_BF16)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(0, 0);
}
#endif

#if defined(DATA_A_IQ1_M)
vec2 get_dm(uint ib, uint a_offset) {
    const uint16_t[4] scales = data_a[a_offset + ib].scales;
    const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
    const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);
    return vec2(d, 0);
}
#endif

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0) || defined(DATA_A_IQ1_S) || defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S) || defined(DATA_A_IQ4_XS) || defined(DATA_A_IQ4_NL) || defined(DATA_A_PLANAR3_0) || defined(DATA_A_ISO3_0) || defined(DATA_A_TQ3_0)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(float(data_a[a_offset + ib].d), 0);
}
#endif

#if defined(DATA_A_Q1_0)
vec2 get_dm(uint ib, uint a_offset) {
    const float d = float(data_a[a_offset + ib].d);
    return vec2(d, 0);
}
#endif

#if defined(DATA_A_MXFP4)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(e8m0_to_fp32(data_a[a_offset + ib].e), 0);
}
#endif

#if defined(DATA_A_NVFP4)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1.0, 0.0);
}
#endif

#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
vec2 get_dm(uint ib, uint a_offset) {
    const vec2 dm = vec2(data_a_packed32[a_offset + ib].dm);
    return dm;
}
#endif

#if defined(DATA_A_PLANAR3_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    uint p = iqs / 2;
    uint j0 = p * 2;
    uint j1 = p * 2 + 1;

    uint b_qs0 = uint(data_a[a_offset + ib].qs[j0 / 4]);
    uint q0 = (b_qs0 >> ((j0 % 4) * 2)) & 0x3;
    uint b_s0 = uint(data_a[a_offset + ib].signs[j0 / 8]);
    uint s0 = (b_s0 >> (j0 % 8)) & 0x1;

    uint b_qs1 = uint(data_a[a_offset + ib].qs[j1 / 4]);
    uint q1 = (b_qs1 >> ((j1 % 4) * 2)) & 0x3;
    uint b_s1 = uint(data_a[a_offset + ib].signs[j1 / 8]);
    uint s1 = (b_s1 >> (j1 % 8)) & 0x1;

    uint cb0 = (p % 64u) * 8u + ((s0 << 2u) | q0);
    uint cb1 = (p % 64u) * 8u + ((s1 << 2u) | q1);

    float r0 = COS_C3B[cb0] + SIN_C3B[cb1];
    float r1 = -SIN_C3B[cb0] + COS_C3B[cb1];

    return vec2(r0, r1);
}

vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    uint p = iqs / 2;

    uint j0 = p * 2;
    uint j1 = p * 2 + 1;

    uint b_qs0 = uint(data_a[a_offset + ib].qs[j0 / 4]);
    uint q0 = (b_qs0 >> ((j0 % 4) * 2)) & 0x3;
    uint b_s0 = uint(data_a[a_offset + ib].signs[j0 / 8]);
    uint s0 = (b_s0 >> (j0 % 8)) & 0x1;

    uint b_qs1 = uint(data_a[a_offset + ib].qs[j1 / 4]);
    uint q1 = (b_qs1 >> ((j1 % 4) * 2)) & 0x3;
    uint b_s1 = uint(data_a[a_offset + ib].signs[j1 / 8]);
    uint s1 = (b_s1 >> (j1 % 8)) & 0x1;

    uint j2 = (p + 1) * 2;
    uint j3 = (p + 1) * 2 + 1;

    uint b_qs2 = uint(data_a[a_offset + ib].qs[j2 / 4]);
    uint q2 = (b_qs2 >> ((j2 % 4) * 2)) & 0x3;
    uint b_s2 = uint(data_a[a_offset + ib].signs[j2 / 8]);
    uint s2 = (b_s2 >> (j2 % 8)) & 0x1;

    uint b_qs3 = uint(data_a[a_offset + ib].qs[j3 / 4]);
    uint q3 = (b_qs3 >> ((j3 % 4) * 2)) & 0x3;
    uint b_s3 = uint(data_a[a_offset + ib].signs[j3 / 8]);
    uint s3 = (b_s3 >> (j3 % 8)) & 0x1;

    uint p0 = p % 64u;
    uint p1 = (p + 1u) % 64u;

    uint cb0 = p0 * 8u + ((s0 << 2u) | q0);
    uint cb1 = p0 * 8u + ((s1 << 2u) | q1);
    uint cb2 = p1 * 8u + ((s2 << 2u) | q2);
    uint cb3 = p1 * 8u + ((s3 << 2u) | q3);

    float r0 = COS_C3B[cb0] + SIN_C3B[cb1];
    float r1 = -SIN_C3B[cb0] + COS_C3B[cb1];
    float r2 = COS_C3B[cb2] + SIN_C3B[cb3];
    float r3 = -SIN_C3B[cb2] + COS_C3B[cb3];

    return vec4(r0, r1, r2, r3);
}
#endif

#if defined(DATA_A_ISO3_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    uint g = iqs / 4;
    uint offset = iqs % 4;

    float qvals[4];
    [[unroll]] for (uint c = 0; c < 4; c++) {
        uint j = g * 4 + c;
        uint b_qs = uint(data_a[a_offset + ib].qs[j / 4]);
        uint qv = (b_qs >> ((j % 4) * 2)) & 0x3;
        uint b_s = uint(data_a[a_offset + ib].signs[j / 8]);
        uint sv = (b_s >> (j % 8)) & 0x1;
        qvals[c] = C3B[(sv << 2) | qv];
    }

    uint qg = g % 32u;
    float qw = PI_QW[qg], qx = -PI_QX[qg], qy = -PI_QY[qg], qz = -PI_QZ[qg];
    float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    float results[4] = {rw, rx, ry, rz};
    return vec2(results[offset], results[offset + 1]);
}

vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    uint g = iqs / 4;

    float qvals[4];
    [[unroll]] for (uint c = 0; c < 4; c++) {
        uint j = g * 4 + c;
        uint b_qs = uint(data_a[a_offset + ib].qs[j / 4]);
        uint qv = (b_qs >> ((j % 4) * 2)) & 0x3;
        uint b_s = uint(data_a[a_offset + ib].signs[j / 8]);
        uint sv = (b_s >> (j % 8)) & 0x1;
        qvals[c] = C3B[(sv << 2) | qv];
    }

    uint qg = g % 32u;
    float qw = PI_QW[qg], qx = -PI_QX[qg], qy = -PI_QY[qg], qz = -PI_QZ[qg];
    float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    return vec4(rw, rx, ry, rz);
}
#endif

#if defined(DATA_A_Q2_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint qsi = (iqs / 64) * 32 + (iqs % 16) * 2; // 0,2,4..30
    const uint scalesi = iqs / 8;                      // 0..15
    const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

    const uvec2 qs = uvec2(data_a[a_offset + ib].qs[qsi], data_a[a_offset + ib].qs[qsi + 1]);
    const uint scales = data_a[a_offset + ib].scales[scalesi];
    const vec2 dm = vec2(data_a[a_offset + ib].dm);

    return dm.x * float(scales & 0xF) * vec2((qs >> qsshift) & 3) - dm.y * float(scales >> 4);
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q3_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                     // 0,1
    const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
    const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
    const uint j = (iqs % 64) / 4;               // 0..3
    const uint is = iqs / 8;                     // 0..15
    const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
    const uint qsshift = halfsplit * 2;          // 0,2,4,6
    const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128

    const int8_t us = int8_t(((data_a[a_offset + ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                          | (((data_a[a_offset + ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
    const float dl = float(data_a[a_offset + ib].d) * float(us - 32);

    return vec2(dl * float(int8_t((data_a[a_offset + ib].qs[qsi    ] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi    ] & m) != 0) ? 0 : 4)),
                dl * float(int8_t((data_a[a_offset + ib].qs[qsi + 1] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi + 1] & m) != 0) ? 0 : 4)));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q4_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

    const vec2 loadd = vec2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uint8_t sc = uint8_t((data_a[a_offset + ib].scales[scidx0] & 0xF) | ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uint8_t mbyte = uint8_t((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0 | ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return vec2(fma(d, float((data_a[a_offset + ib].qs[qsi    ] >> (b * 4)) & 0xF), m),
                fma(d, float((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF), m));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q5_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
    const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

    const uint8_t hm = uint8_t(1 << (iqs / 16));

    const vec2 loadd = vec2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uint8_t sc    = uint8_t((data_a[a_offset + ib].scales[scidx0] & 0xF)                         | ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uint8_t mbyte = uint8_t(((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0) | ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return vec2(fma(d, float((data_a[a_offset + ib].qs[qsi    ] >> (b * 4)) & 0xF) + float((data_a[a_offset + ib].qh[qhi    ] & hm) != 0 ? 16 : 0), m),
                fma(d, float((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF) + float((data_a[a_offset + ib].qh[qhi + 1] & hm) != 0 ? 16 : 0), m));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q6_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                    // 0,1
    const uint b = (iqs % 64) / 32;             // 0,1
    const uint is_b = (iqs % 16) / 8;           // 0,1
    const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
    const uint is = 8 * n + qhshift + is_b;     // 0..15
    const uint qsi = n * 64 + (iqs % 32) * 2;   // 0,2,4..126
    const uint qhi = n * 32 + (iqs % 16) * 2;   // 0,2,4..62

    const float dscale = float(data_a[a_offset + ib].d) * float(data_a[a_offset + ib].scales[is]);

    return vec2(dscale * float(int8_t(((data_a[a_offset + ib].ql[qsi    ] >> (b * 4)) & 0xF) | (((data_a[a_offset + ib].qh[qhi    ] >> qhshift) & 3) << 4)) - 32),
                dscale * float(int8_t(((data_a[a_offset + ib].ql[qsi + 1] >> (b * 4)) & 0xF) | (((data_a[a_offset + ib].qh[qhi + 1] >> qhshift) & 3) << 4)) - 32));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif
