CXX = g++
CPPFLAGS = -g -Wall -O3 -std=c++11
LDFLAGS = -g
LDLIBS = -lm -larmadillo -lopencv_core -lopencv_imgproc -lopencv_highgui

SRCS = ../src/main.cpp ../src/simulate.cpp ../src/lmbTracker.cpp \
       ../src/bestAssignments.cpp ../src/shortestPaths.cpp 
       
OBJS = $(subst .cpp,.o,$(SRCS))

all: exec

exec: $(OBJS)
	$(CXX) $(LDFLAGS) -oexec.out $(OBJS) $(LDLIBS)
	
depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;
	
clean:
	rm -f $(OBJS) *.out
	
dist-clean:
	rm -f *~ .depend
	
include .depend	
