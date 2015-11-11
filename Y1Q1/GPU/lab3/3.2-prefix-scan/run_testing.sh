#!/bin/bash
echo 'Number of passed tests: ' 
more testing.log | grep 'TEST PASSED' | wc -l
echo 'Number of failed tests: ' 
more testing.log | grep 'TEST FAILED' | wc -l

