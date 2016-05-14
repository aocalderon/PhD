"""Custom topology example

Two directly connected switches plus a host for each switch:

   host --- switch --- switch --- host

Adding the 'topos' dict with a key/value pair to generate our newly defined
topology enables one to pass in '--topo=mytopo' from the command line.
"""

from mininet.topo import Topo

class MyTopo2( Topo ):
    "Simple topology example."

    def __init__( self ):
        "Create custom topo."

        # Initialize topology
        Topo.__init__( self )

        # Add hosts and switches
        leftUpHost = self.addHost( 'h1', ip="10.0.0.1/16")
        leftDownHost = self.addHost( 'h2', ip="10.0.0.2/16")
        leftSwitch = self.addSwitch( 's1' )
        middleSwitch = self.addSwitch( 's2' )
        rightSwitch = self.addSwitch( 's3' )
        rightUpHost = self.addHost( 'h3', ip="10.0.1.1/16")
        rightMiddleHost = self.addHost( 'h4', ip="10.0.1.2/16")
        rightDownHost = self.addHost( 'h5', ip="10.0.1.3/16")

        # Add links
        self.addLink( leftUpHost, leftSwitch )
        self.addLink( leftDownHost, leftSwitch )
	self.addLink( leftSwitch, middleSwitch )
	self.addLink( middleSwitch, rightSwitch )
        self.addLink( rightSwitch, rightUpHost )
        self.addLink( rightSwitch, rightMiddleHost )
        self.addLink( rightSwitch, rightDownHost )


topos = { 'mytopo2': ( lambda: MyTopo2() ) }
