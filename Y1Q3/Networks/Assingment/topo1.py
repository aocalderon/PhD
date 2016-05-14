from mininet.topo import Topo

class MyTopo2( Topo ):
    "Simple topology example."

    def __init__( self ):
        "Create custom topo."

        # Initialize topology
        Topo.__init__( self )

        # Add hosts and switches
        leftUpHost = self.addHost( 'h1')
        leftDownHost = self.addHost( 'h2')
        leftSwitch = self.addSwitch( 's1' )
        middleSwitch = self.addSwitch( 's2' )
        rightSwitch = self.addSwitch( 's3' )
        rightUpHost = self.addHost( 'h3')
        rightMiddleHost = self.addHost( 'h4')
        rightDownHost = self.addHost( 'h5')

        # Add links
        self.addLink( leftUpHost, leftSwitch )
        self.addLink( leftDownHost, leftSwitch )
		self.addLink( leftSwitch, middleSwitch )
		self.addLink( middleSwitch, rightSwitch )
        self.addLink( rightSwitch, rightUpHost )
        self.addLink( rightSwitch, rightMiddleHost )
        self.addLink( rightSwitch, rightDownHost )


topos = { 'mytopo2': ( lambda: MyTopo2() ) }
