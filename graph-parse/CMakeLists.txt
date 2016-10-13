PROJECT(cnn:graph-parse)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(graph-parse-new-cost graph-parse-new-cost.cc)
target_link_libraries(graph-parse-new-cost cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(graph-parse-new-cost-broken graph-parse-new-cost-broken.cc)
target_link_libraries(graph-parse-new-cost-broken cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(graph-parse-new-cost-mbr graph-parse-new-cost-mbr.cc)
target_link_libraries(graph-parse-new-cost-mbr cnn ${Boost_LIBRARIES})

