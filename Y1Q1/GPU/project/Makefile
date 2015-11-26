#
# Makefile for example programs
#

subdir = /opt/PostgerSQL/postgresql-9.4.5/src/test/examples
top_builddir = /opt/PostgreSQL/postgresql-9.4.5
include $(top_builddir)/src/Makefile.global

ifeq ($(PORTNAME), win32)
LDLIBS += -lws2_32
endif

override CPPFLAGS := -I$(libpq_srcdir) $(CPPFLAGS)
override LDLIBS := $(libpq_pgport) $(LDLIBS)


PROGS = test

all: $(PROGS)

clean:
	rm -f $(PROGS)
