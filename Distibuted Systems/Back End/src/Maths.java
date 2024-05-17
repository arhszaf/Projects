package src;

import static java.lang.Math.*;
import java.util.ArrayList;
import java.util.List;



public class Maths {
    public static void main(String[] args){
        List<String> list = new ArrayList<>();
        list.add("37.95001155239993");//0
        list.add("23.69503479744284");//1
        list.add("37.949571620793144");//2
        list.add("23.695699985278534");//3


        //Double processedValue = acos(sin(list.get(1))*sin(list.get(3)*sin(list.get(5))*sin(list.get(7))*sin(list.get(8)))+cos(list.get(0))*cos(list.get(2))*cos(list.get(3)-list.get(1)))*6371;
        //usefull code
        double processedValue = acos(sin(Double.parseDouble(list.get(0)))*sin(Double.parseDouble(list.get(2)))+cos(Double.parseDouble(list.get(0)))*cos(Double.parseDouble(list.get(2)))*cos(Double.parseDouble(list.get(3))-Double.parseDouble(list.get(1))))*6371;
        System.out.println(processedValue);
        /*  Basic Idea: intermediate results
        *   first(lat1,lon1,lat2,lon2) second(lat2,lon2,lat3,lon3)....
        * Implementation:
        *   1. with a for loop
        *   2. with Streams?
        *
        * Basic Idea:synchronization
        *
        * Basic Idea:
        * */

    }
}
//  [lat, 37.94715194986727], [lon, 23.694605644000458],
//  , [lat, 37.946653163250694], [lon, 23.695273887184726], [time, 2023-03-19T17:39:40Z],
//  [lat, 37.94606938059409], [lon, 23.696046363381015],
//  [lat, 37.94524023203743], [lon, 23.695102225807773], [lat, 37.944461839163594],
//  [lon, 23.694050799873935], [lat, 37.943697486759284], [lon, 23.69324249390282],
//  [lat, 37.943405584269506], [lon, 23.69371456268944],[lat, 37.943189829510054],
//  [lon, 23.694036427771227], [lat, 37.942627171961284],
//  [lon, 23.694884005819933],[lat, 37.94196891990529],
//  [lon, 23.695816839239242], [time, 2023-03-19T17:44:36Z], [lat, 37.941774313796756],
//  [lon, 23.695747101804855],  [lat, 37.941499326025664], [lon, 23.695532525083664],
//  [lat, 37.94130471867361], [lon, 23.695323312780502],  [lat, 37.94097984389345],
//  [lon, 23.694886713748932],  [lat, 37.94045524527583], [lon, 23.69421079707718],
//  [lat, 37.94014217657466], [lon, 23.693808465724945],  [lat, 37.93978679967615],
//  [lon, 23.693347125774384],  [lat, 37.93952161016134], [lon, 23.69299274828799],
//  [lat, 37.93923603716365], [lon, 23.692619921234918],
//  [lat, 37.9387847281908], [lon, 23.692015802071865],  [lat, 37.93887357378787],
//  [lon, 23.691527640031154],[lat, 37.93877203595387], [lon, 23.691431080506618],
//  [lat, 37.93911472558143], [lon, 23.690717612908657],  [lat, 37.939673179328615],
//  [lon, 23.690224610251718], [lat, 37.94003701812341], [lon, 23.690106593055063],
//  [lat, 37.94053200521952], [lon, 23.6907074078744], , [lat, 37.94081968849565],
//  [lon, 23.69096489993983],  [lat, 37.94162423069252],
//  [lon, 23.69053836283811],  [lat, 37.942208048664426], [lon, 23.691133813239418], ,
//  [lat, 37.94297477498236], [lon, 23.69215686087734],  [lat, 37.94371087900379],
//  [lon, 23.69313854937679], [lat, 37.944477889945155], [lon, 23.693989646908282],
//  [lat, 37.94513360173761], [lon, 23.693201077457903]
