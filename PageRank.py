import re
import sys
from operator import add

from pyspark import SparkConf
from pyspark.sql import SparkSession


def parseNeighbors(line):
    data = line.split('\t')
    source = data[1]
    content = data[3]
    pattern = re.compile("<target>(.*?)</target>")
    result = pattern.findall(content)
    return (source, result)

def constructPair(line):
    for dest in line[1]:
        yield (line[0], dest)

def calCell(pair):
    if len(pair) < 2:
        return
    dests = pair[0]
    rank = pair[1]
    num = len(dests)
    for dest in dests:
        yield (dest, rank / num)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Parameters Error: Expect 3, Given " + str(len(sys.argv)))
        exit(-1)

    spark = SparkSession \
        .builder \
        .appName("PageRank") \
        .getOrCreate()

    sc = spark.sparkContext

    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])

    #print("\n\n\n\nNumber of lines: " + str(lines.count()) + "\n\n")

    lines = lines.map(lambda r : parseNeighbors(r))

    #print("\n\n\n\nNumber of Records: " + str(lines.count()) + "\n\n")

    tokens = lines.flatMap(lambda data: [(data[0], x) for x in data[1]])

    #print("\n\n\n\nNumber of From_To Pairs: " + str(tokens.count()) + "\n\n")

    tokens = tokens.distinct().groupByKey().cache()

    #print("\n\n\n\nNumber of Tokens: " + str(tokens.count()) + "\n\n")

    ranks = tokens.map(lambda pair: (pair[0], 1.0))

    #print("\n\n\n\nNumber of Ranks initialized: " + str(ranks.count()) + "\n\n")

    for itr in range(int(sys.argv[4])):
        cell = tokens.join(ranks).flatMap(
            lambda pair : calCell(pair[1])
        )

        #print("\n\n\n\nNumber of Cells: " + str(cell.count()) + "\t" + str(itr) + " Stage of Iteration Complete.\n\n")
        #print("\n\n\n\nNumber of Tokens: " + str(tokens.count()) + "\t" + str(itr) + " Stage of Iteration Complete.\n\n")


        ranks = cell.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)

        #print("\n\n\n\nNumber of Ranks: " + str(ranks.count()) + "\t" + str(itr) + " Stage of Iteration Complete.\n\n")

    res = ranks.takeOrdered(sys.argv[3], key=lambda x: -x[1])

    res = sc.parallelize(res)
    #print(ranks.takeOrdered(10, key=lambda x: -x[1]).__class__)
    #ranks.sortBy(lambda x: x[1], False).saveAsTextFile(sys.argv[3])
    res.saveAsTextFile(sys.argv[2])


    #print("Page: " + results[0] + "\tProbability: " + str(results[1]) + "\tRank: " + str(1))

    '''
    sortedRanks = ranks.sortBy(lambda a : a[1])

    for i in range(100):
        print("Page: " + sortedRanks[0-i][0] + "\tProbability: " + str(sortedRanks[0-i][1]) + "\tRank: " + str(i))
    '''

    spark.stop()


