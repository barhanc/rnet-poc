import { Text, View, StyleSheet } from "react-native";
import {
  answerToTheUltimateQuestionOfLifeTheUniverseAndEverything,
  isWednesday,
  myAwesomeArray,
  giveMeFive,
  sumMeThis,
  divideMeThis,
  reverseMeThis,
  sumMeThisObject,
  sumMeThisArray,
  nativeMap,
  runJsFunction,
  getDateObject,
  getInfinityObject,
  checkExecuTorch,
} from "react-native-my-lib";

import { useEffect } from "react";

export default function App() {
  useEffect(() => {
    runJsFunction();

    const date = getDateObject();
    console.log(date);
    console.log(date.day, date.month, date.year);
    console.log(date.now);
    console.log(date.hello);
    console.log("------------------");
    date.day = 5;
    date.month = 12;
    date.year = 2025;
    console.log(date.day, date.month, date.year);
    console.log("------------------");

    const infinityObject = getInfinityObject();
    console.log(infinityObject.a);
    console.log(infinityObject.wgwrw);
    console.log(infinityObject.jboqo345vj);
    infinityObject.x = 0;
    infinityObject.x = 0;
    console.log(infinityObject.ojbob);
  });

  return (
    <View style={styles.container}>
      <Text>
        answerToTheUltimateQuestionOfLifeTheUniverseAndEverything:
        {answerToTheUltimateQuestionOfLifeTheUniverseAndEverything}
      </Text>
      <Text>isWednesday:{isWednesday.toString()}</Text>
      <Text>myAwesomeArray:{myAwesomeArray.toString()}</Text>
      <Text>giveMeFive:{giveMeFive()}</Text>
      <Text>sumMeThis:{sumMeThis(5, 2)}</Text>
      <Text>divideMeThis:{divideMeThis(5, 2)}</Text>
      <Text>reverseMeThis:{reverseMeThis("Hello, world!")}</Text>
      <Text>sumMeThisObject:{JSON.stringify(sumMeThisObject({ firstNum: 5, lastNum: 2 }))}</Text>
      <Text>sumMeThisArray:{sumMeThisArray([1, 2, 3, 4, 5])}</Text>
      <Text>nativeMap:{nativeMap([1, 2, 3, 4, 5], (x) => x * x).join(", ")}</Text>
      <Text>runJsFunction:Check console logs</Text>
      <Text>checkExecuTorch:{checkExecuTorch()}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
});
