import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 512
    height: 512

    ColumnLayout {
        width: parent.width

        Rectangle {
            Layout.fillWidth: true
            height: 128
            color: "dodgerblue"
        }

        Rectangle {
            width: 256
            height: 256
            color: "transparent"

            ScrollView {
                width: 256
                height: 256
                clip: true

                Image {
                    width: 1024
                    height: 1024
                    source: "../images/blueberry.jpg"
                }

            }

        }

    }

}
