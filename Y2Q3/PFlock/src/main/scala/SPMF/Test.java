package SPMF;
import com.jcraft.jsch.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;


public class Test {
    public static void main(String[] arg) throws JSchException, SftpException, FileNotFoundException {
        JSch jsch = new JSch();
        Session session;
        session = jsch.getSession("acald013", "bolt.cs.ucr.edu", 22);

        String privateKey = "~/.ssh/dblab";
        jsch.addIdentity(privateKey);
        session.setConfig("StrictHostKeyChecking", "no");
        session.connect();
        ChannelSftp channel;
        channel = (ChannelSftp) session.openChannel("sftp");
        channel.connect();
        File localFile = new File("/tmp/test.txt");
        //If you want you can change the directory using the following line.
        channel.cd("/home/csgrads/acald013/");
        channel.put(new FileInputStream(localFile), localFile.getName());
        channel.disconnect();
        session.disconnect();
    }
}