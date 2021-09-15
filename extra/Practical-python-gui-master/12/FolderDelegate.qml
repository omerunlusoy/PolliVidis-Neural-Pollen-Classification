import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

Component {
    Rectangle {

        id: folderDele
        width: 200
        height: 200
        color: folderColor

        ColumnLayout {
            width: parent.width
            height: parent.height

            Image {
                anchors.centerIn: parent
                sourceSize.width: 128
                sourceSize.height: 128
                source: folderImage
            }

            Text {
                padding: 8
                anchors.bottom: parent.bottom
                text: folderName
                color: "white"
                font.pixelSize: 20
            }
        }

        MouseArea {
            anchors.fill: parent
            hoverEnabled: true

            onEntered: {
                var x = parent.x
                var y = parent.y
                var index = gView.indexAt(x, y)
                gView.currentIndex = index

            }

            onPressed: {
                if(folderName === 'Music') {
                    gView.model.setProperty(gView.currentIndex, 'folderName', 'Music Changed')
                } else {
                    gView.model.setProperty(gView.currentIndex, 'folderName', 'changed')
                }
            }

        }

    }
}
