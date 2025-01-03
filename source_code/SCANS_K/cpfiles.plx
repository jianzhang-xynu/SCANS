#! /usr/bin/perl

open LIST,"lists/test.list";
chomp(@list=<LIST>);
close LIST;

foreach $list(@list){
	system "cp feas/CS_K/PCfeas/$list.txt feas/PCfeas/$list.txt";
	system "cp feas/CS_K/PSSMfeas/$list.pssm feas/PSSMfeas/$list.pssm";
	system "cp seqs/CS_K/$list.txt seqs/$list.txt"
}

